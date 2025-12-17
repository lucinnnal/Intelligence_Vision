import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import AutoModel, CLIPVisionModel, Dinov2Model, AutoImageProcessor
import wandb

# --------------------------------------------------------------------------- 
# 1. SETUP & CONSTANTS
# --------------------------------------------------------------------------- 

# Add MFM directory to path for local imports
mfm_path = os.path.join(os.path.dirname(__file__), 'MFM')
if mfm_path not in sys.path:
    sys.path.insert(0, mfm_path)

try:
    from config import get_custom_config, _C
    from models.build import build_model
    MFM_AVAILABLE = True
except ImportError:
    print("Warning: MFM library not found. MFM-related models will not be available.")
    MFM_AVAILABLE = False

GENERATOR_CLASSES = [
    'ADM', 'DDPM', 'Diff-ProjectedGAN', 'Diff-StyleGAN2', 'IDDPM',
    'LDM', 'PNDM', 'ProGAN', 'ProjectedGAN', 'StyleGAN'
]

# --------------------------------------------------------------------------- 
# 2. DATASET
# --------------------------------------------------------------------------- 

class GenerativeImageDataset(Dataset):
    """
    Dataset for loading images from different generator classes.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} does not exist.")
            return

        print(f"Scanning {split} dataset in {split_dir}...")
        for class_idx, class_name in enumerate(GENERATOR_CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = 0
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        count += 1
                print(f"  [{class_name}] found {count} images")
        print(f"Total {split} images: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning a random image instead.")
            # Return a random image to prevent training crash
            random_idx = torch.randint(0, len(self.samples), (1,)).item()
            return self.__getitem__(random_idx)

# --------------------------------------------------------------------------- 
# 3. MODEL COMPONENTS
# --------------------------------------------------------------------------- 

class MFMEncoder(nn.Module):
    """MFM ViT-Base pretrained model as a feature extractor."""
    def __init__(self, pretrained_path):
        super().__init__()
        if not MFM_AVAILABLE:
            raise ImportError("MFM library is required for MFMEncoder.")

        mfm_config = _C.clone()
        mfm_config.MODEL.TYPE = 'vit'
        mfm_config.MODEL.VIT.PATCH_SIZE = 16
        mfm_config.MODEL.VIT.EMBED_DIM = 768
        mfm_config.MODEL.VIT.DEPTH = 12
        mfm_config.MODEL.VIT.NUM_HEADS = 12
        mfm_config.freeze()

        mfm_full_model = build_model(mfm_config, is_pretrain=True)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading MFM weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            mfm_full_model.load_state_dict(state_dict, strict=False)

        self.encoder = mfm_full_model.encoder
        self.feature_dim = self.encoder.embed_dim

    def forward(self, x):
        return self.encoder(x, x)

class VisionEncoder(nn.Module):
    """Generic Vision Encoder for DINO or CLIP."""
    def __init__(self, model_name, encoder_type):
        super().__init__()
        print(f"Loading {encoder_type.upper()} model: {model_name}")
        if encoder_type.lower() == 'dino':
            self.model = AutoModel.from_pretrained(model_name)
        elif encoder_type.lower() == 'clip':
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown vision encoder type: {encoder_type}")

        self.encoder_type = encoder_type.lower()
        self.feature_dim = self.model.config.hidden_size

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        if self.encoder_type == 'clip':
            # Use pooler_output for CLIP
            return outputs.pooler_output
        else:
            # Use CLS token for DINO
            return outputs.last_hidden_state[:, 0, :]

class MLPClassifier(nn.Module):
    """A simple MLP classifier head."""
    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

# --------------------------------------------------------------------------- 
# 4. UNIFIED MODEL
# --------------------------------------------------------------------------- 

class UnifiedModel(nn.Module):
    """
    A single model that can be configured to run any of the experiments.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        self.model_type = model_config['type']
        
        self.mfm_encoder = None
        self.vision_encoder = None
        
        # --- Build Backbones ---
        if 'mfm' in self.model_type:
            if not MFM_AVAILABLE:
                raise RuntimeError("MFM model specified in config, but MFM library is not available.")
            self.mfm_encoder = MFMEncoder(config['mfm_pretrained_path'])
        
        if 'clip' in self.model_type or 'dino' in self.model_type:
            encoder_type = 'clip' if 'clip' in self.model_type else 'dino'
            self.vision_encoder = VisionEncoder(model_config['vision_model_name'], encoder_type)
        
        # --- Build Classifier Head ---
        input_dim = self._get_classifier_input_dim()
        self.classifier = MLPClassifier(input_dim, config['num_classes'], model_config['hidden_dim'])

    def _get_classifier_input_dim(self):
        dim = 0
        if self.mfm_encoder:
            dim += self.mfm_encoder.feature_dim
        if self.vision_encoder:
            dim += self.vision_encoder.feature_dim
        if dim == 0:
            raise ValueError("Model misconfiguration: at least one encoder must be specified.")
        return dim

    def forward(self, x):
        features = []
        if self.mfm_encoder:
            mfm_feat = self.mfm_encoder(x)
            # MFM returns [B, N, D], take the CLS token
            features.append(mfm_feat[:, 0, :])
        
        if self.vision_encoder:
            vision_feat = self.vision_encoder(x)
            features.append(vision_feat)
        
        # --- Feature Fusion ---
        if len(features) > 1:
            fused_features = torch.cat(features, dim=1)
        else:
            fused_features = features[0]
            
        logits = self.classifier(fused_features)
        return logits

    def get_params_to_train(self):
        """Returns a list of parameters to be trained based on the config."""
        model_config = self.config['model']
        params = list(self.classifier.parameters())
        print("Training parameters: Classifier")

        if self.mfm_encoder and model_config.get('train_mfm_encoder', False):
            params.extend(list(self.mfm_encoder.parameters()))
            print("Training parameters: MFM Encoder")
        elif self.mfm_encoder:
            for param in self.mfm_encoder.parameters():
                param.requires_grad = False
            print("Freezing parameters: MFM Encoder")

        if self.vision_encoder and model_config.get('train_vision_encoder', False):
            params.extend(list(self.vision_encoder.parameters()))
            print(f"Training parameters: {self.vision_encoder.encoder_type.upper()} Encoder")
        elif self.vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print(f"Freezing parameters: {self.vision_encoder.encoder_type.upper()} Encoder")
        
        return params

# NOTE: Prompt-tuning models are highly specialized and have been excluded for now
# to create a clean, unified fine-tuning script first. If prompt tuning is still
# required, it can be added back as a separate, more complex model class.

# --------------------------------------------------------------------------- 
# 5. TRAINER
# --------------------------------------------------------------------------- 

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # --- WandB ---
        if config.get('use_wandb', False):
            wandb.init(project=config['project_name'], config=config, name=config['run_name'])
        
        # --- Transforms ---
        norm_config = config['normalization']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_config['mean'], std=norm_config['std'])
        ])

        # --- DataLoaders ---
        self._init_dataloaders()

        # --- Model ---
        self.model = UnifiedModel(config).to(self.device)
        if config.get('use_wandb', False):
            wandb.watch(self.model)

        # --- Optimizer & Scheduler ---
        self._init_optimizer()
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        os.makedirs(config['output_dir'], exist_ok=True)
        self.best_val_metric = 0

    def _init_dataloaders(self):
        train_dataset = GenerativeImageDataset(self.config['data_root'], 'train', self.transform)
        val_dataset = GenerativeImageDataset(self.config['data_root'], 'val', self.transform)
        self.test_dataset = GenerativeImageDataset(self.config['data_root'], 'test', self.transform)
        
        bs = self.config['batch_size']
        nw = self.config.get('num_workers', 4)
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
        self.val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw)
        self.test_loader = DataLoader(self.test_dataset, batch_size=bs, shuffle=False, num_workers=nw)

    def _init_optimizer(self):
        params_to_train = self.model.get_params_to_train()
        opt_config = self.config['optimizer']
        opt_class = getattr(optim, opt_config['name'])
        self.optimizer = opt_class(filter(lambda p: p.requires_grad, params_to_train), **opt_config['params'])

        if 'scheduler' in self.config:
            sched_config = self.config['scheduler']
            sched_class = getattr(optim.lr_scheduler, sched_config['name'])
            self.scheduler = sched_class(self.optimizer, **sched_config['params'])
        else:
            self.scheduler = None
        print(f"\nOptimizer: {opt_config['name']}, Scheduler: {self.config.get('scheduler', {}).get('name', 'None')}\n")

    def train(self):
        print("--- Starting Training ---")
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc, preds, labels = self._validate()

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.config['optimizer']['params']['lr']
                })

            if val_acc > self.best_val_metric:
                self.best_val_metric = val_acc
                self.save_model('best_model.pth')
                print(f"--> New best model saved with Val Acc: {val_acc:.2f}%")
                if self.config.get('use_wandb', False):
                    wandb.run.summary["best_val_accuracy"] = val_acc
            
            if self.scheduler:
                self.scheduler.step()
        print(f"\n--- Training Finished. Best Val Acc: {self.best_val_metric:.2f}% ---")

    def _train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*total_correct/total_samples:.2f}%'})
        return total_loss / total_samples, 100. * total_correct / total_samples

    def _validate(self, loader=None):
        self.model.eval()
        current_loader = loader or self.val_loader
        desc = "Validating" if loader is None else "Testing"
        
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(current_loader, desc=desc):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / total_samples
        accuracy = 100. * total_correct / total_samples
        return avg_loss, accuracy, all_preds, all_labels

    def test(self):
        print("\n--- Starting Testing with the Best Model ---")
        try:
            self.load_model('best_model.pth')
        except FileNotFoundError:
            print("Best model not found. Testing is skipped.")
            return

        test_loss, test_acc, test_preds, test_labels = self._validate(loader=self.test_loader)
        print(f"\nTest Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
        
        report = classification_report(test_labels, test_preds, target_names=GENERATOR_CLASSES, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=GENERATOR_CLASSES))
        
        if self.config.get('use_wandb', False):
            wandb.log({"test_accuracy": test_acc, "test_loss": test_loss, "classification_report": report})
            wandb.finish()

    def save_model(self, filename):
        path = os.path.join(self.config['output_dir'], filename)
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename):
        path = os.path.join(self.config['output_dir'], filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

# --------------------------------------------------------------------------- 
# 6. MAIN EXECUTION
# --------------------------------------------------------------------------- 

def main():
    parser = argparse.ArgumentParser(description="Unified model training script for generative model classification.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        sys.exit(1)

    # For simplicity, prompt-tuning models are not included in this unified script.
    if 'prompt' in config['model']['type']:
        print("="*80)
        print("WARNING: This unified script does not support prompt-tuning models.")
        print("The original prompt-tuning scripts were complex and have been separated.")
        print("Please run 'clip_prompt.py' or 'mfm_clip_prompt.py' for those experiments.")
        print("="*80)
        # We will keep the original files for this reason.
        return

    trainer = Trainer(config)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()
