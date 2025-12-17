import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPVisionModel

# --------------------------------------------------------------------------- 
# 1. 설정 및 경로
# --------------------------------------------------------------------------- 

# 10개 생성 모델 클래스
GENERATOR_CLASSES = [
    'ADM', 'DDPM', 'Diff-ProjectedGAN', 'Diff-StyleGAN2', 'IDDPM',
    'LDM', 'PNDM', 'ProGAN', 'ProjectedGAN', 'StyleGAN'
]

# --------------------------------------------------------------------------- 
# 2. 통합 모델 정의 (Deep Prompt Tuning 적용)
# --------------------------------------------------------------------------- 

class CLIPPromptTuningModel(nn.Module):
    """CLIP Vision Encoder에 Deep Prompt Tuning을 적용한 모델"""
    def __init__(self, config):
        super().__init__()
        print("\nCLIP Deep Prompt Tuning 모델 초기화 중...")
        self.config = config
        self.prompt_config = config.get('prompt_config', {})
        self.use_deep_prompt = self.prompt_config.get('use_deep_prompt', False)

        # --- 백본 모델 로드 ---
        self.vision_encoder_backbone = self._build_vision_backbone()

        # --- Deep Prompt 파라미터 생성 ---
        if self.use_deep_prompt:
            print("Deep Prompt Tuning이 활성화되었습니다.")
            prompt_len = self.prompt_config.get('prompt_len', 10)
            
            # Vision Encoder Prompts
            vision_dim = self.vision_encoder_backbone.config.hidden_size
            vision_layers = self.vision_encoder_backbone.config.num_hidden_layers
            self.vision_prompts = nn.Parameter(torch.zeros(vision_layers, prompt_len, vision_dim))
            nn.init.xavier_uniform_(self.vision_prompts)
            print(f" - Vision Prompts 생성: {vision_layers} layers, {prompt_len} tokens, {vision_dim} dim")

        # --- 최종 분류기 ---
        vision_feature_dim = self.vision_encoder_backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(vision_feature_dim),
            nn.Linear(vision_feature_dim, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config['hidden_dim'], config['num_classes'])
        )

    def _build_vision_backbone(self):
        encoder_type = self.config.get('vision_encoder', 'clip')
        if encoder_type != 'clip':
             raise ValueError("이 모델은 'clip' vision encoder만 지원합니다.")

        model_name = self.config[f'{encoder_type}_model_name']
        print(f"{encoder_type.upper()} 모델 로드: {model_name}")
        
        return CLIPVisionModel.from_pretrained(model_name)


    def forward_vision_with_prompt(self, x):
        model = self.vision_encoder_backbone
        B = x.shape[0]

        # 1. 임베딩
        embedding_output = model.vision_model.embeddings(x)
        num_tokens = embedding_output.shape[1]
        
        hidden_state = model.vision_model.pre_layrnorm(embedding_output)

        # 2. Deep Prompt 주입을 위한 마스크 생성
        prompt_len = self.prompt_config.get('prompt_len', 10)
        prompted_seq_len = num_tokens + prompt_len
        # Vision 모델에서는 모든 토큰이 서로를 참조해야 하므로, 0으로 채워진 마스크를 생성하여 전달합니다.
        causal_attention_mask = torch.zeros(
            (B, 1, prompted_seq_len, prompted_seq_len),
            dtype=hidden_state.dtype,
            device=hidden_state.device
        )

        # 3. Deep Prompt 주입 루프
        for i, layer_module in enumerate(model.vision_model.encoder.layers):
            prompt = self.vision_prompts[i].unsqueeze(0).expand(B, -1, -1)
            input_prompted = torch.cat((hidden_state, prompt), dim=1)
            
            layer_outputs = layer_module(
                input_prompted,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask
            )

            hidden_state_prompted = layer_outputs[0]
            hidden_state = hidden_state_prompted[:, :num_tokens, :]
        
        # 4. 최종 Layer Normalization 적용
        sequence_output = model.vision_model.post_layernorm(hidden_state)

        return sequence_output[:, 0] # CLS 토큰 반환

    def forward(self, x):
        if self.use_deep_prompt:
            vision_feat = self.forward_vision_with_prompt(x)
        else:
            # 프롬프트 미사용 시, 기존 방식 (백본 fine-tuning)
            vision_outputs = self.vision_encoder_backbone(pixel_values=x)
            vision_feat = vision_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(vision_feat)
        return logits


# --------------------------------------------------------------------------- 
# 3. 데이터셋
# --------------------------------------------------------------------------- 
class GenerativeImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir): raise FileNotFoundError(f"경로를 찾을 수 없습니다: {split_dir}")
        for class_idx, class_name in enumerate(GENERATOR_CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), class_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"이미지 로딩 오류 {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.long)

# --------------------------------------------------------------------------- 
# 4. Trainer
# --------------------------------------------------------------------------- 
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._init_dataloaders()
        self._init_model()
        self._init_optimizer()

    def _init_dataloaders(self):
        print("\n데이터셋 로드 중...")
        self.train_dataset = GenerativeImageDataset(self.config['data_root'], 'train', self.transform)
        self.val_dataset = GenerativeImageDataset(self.config['data_root'], 'val', self.transform)
        self.test_dataset = GenerativeImageDataset(self.config['data_root'], 'test', self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        print(f"데이터셋 크기 - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def _init_model(self):
        self.model = CLIPPromptTuningModel(self.config).to(self.device)

    def _init_optimizer(self):
        print("\n옵티마이저 설정:")
        params_to_train = []
        
        prompt_config = self.config.get('prompt_config', {})
        if prompt_config.get('use_deep_prompt', False):
            # Deep Prompt Tuning: 프롬프트와 분류기만 학습
            params_to_train.append(self.model.vision_prompts)
            params_to_train.extend(self.model.classifier.parameters())
            
            # 백본 모델 고정
            for param in self.model.vision_encoder_backbone.parameters():
                param.requires_grad = False
            
            print(" - Deep Prompts (Vision)와 Classifier 파라미터만 학습합니다.")

        else:
            # 기존 Fine-tuning 방식
            params_to_train.extend(self.model.classifier.parameters())
            print(" - Classifier 파라미터가 학습에 추가됩니다.")
            
            if self.config.get('train_vision_encoder', False):
                params_to_train.extend(self.model.vision_encoder_backbone.parameters())
                print(" - Vision Encoder 파라미터가 학습에 추가됩니다. (Fine-tuning)")
            else:
                for param in self.model.vision_encoder_backbone.parameters():
                    param.requires_grad = False

        opt_name = self.config['optimizer']
        opt_params = self.config['optimizer_params']
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params_to_train), **opt_params)

        # 스케줄러 설정
        sched_name = self.config['scheduler']
        sched_params = self.config['scheduler_params']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **sched_params) if sched_name.lower() == 'cosineannealinglr' else None
        self.criterion = nn.CrossEntropyLoss()

    def train(self, patience=7):
        best_val_loss = float('inf')
        patience_counter = 0
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, 'best_model_prompt.pth')

        for epoch in range(self.config['num_epochs']):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')
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
                pbar.set_postfix({'loss': f'{total_loss/total_samples:.4f}', 'acc': f'{100.*total_correct/total_samples:.2f}%'})
            
            if self.scheduler: self.scheduler.step()

            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 학습되는 파라미터만 저장
                torch.save(self.model.state_dict(), model_save_path)
                print(f"--> 최고 모델 저장됨: {model_save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("조기 종료!")
                    break

    def validate(self, loader=None):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        current_loader = loader or self.val_loader
        with torch.no_grad():
            for images, labels in tqdm(current_loader, desc='Validating', leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)
        return total_loss / total_samples, 100. * total_correct / total_samples

    def test(self):
        model_path = os.path.join(self.config['output_dir'], 'best_model_prompt.pth')
        if not os.path.exists(model_path):
            print("저장된 모델을 찾을 수 없습니다.")
            return

        print(f"\n최고 모델 테스트: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("\n테스트 결과:")
        print(classification_report(all_labels, all_preds, target_names=GENERATOR_CLASSES))
        print(f"전체 정확도: {accuracy_score(all_labels, all_preds) * 100:.2f}%")


import argparse
import sys

# ... (rest of the imports)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP Deep Prompt-Tuning script.")
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
        
    trainer = Trainer(config)
    trainer.train()
    trainer.test()
