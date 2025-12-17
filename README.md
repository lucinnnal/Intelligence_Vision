# Experiment Execution Guide

This project was designed to be executed in a **GPU-enabled environment**. To ensure reproducibility and ease of setup, **all experiments were conducted using Google Colab via `main.ipynb`**.

---

## 🚀 Recommended: Run on Google Colab (GPU)

### Dataset

- The `data.zip` file can be downloaded from **Google Drive**:  
  👉 [**data.zip**](https://drive.google.com/file/d/10b1yPJOOsW2nCoZcNUPlVIDjccWcax_o/view?usp=drive_link)

### Pretrained Weights

- The **MFM_pretrained_vit_base** weights are available on **Google Drive**:  
  👉 [**MFM_pretrained_vit_base**](https://drive.google.com/file/d/1Y1q6E3LmkPfXoywYdRPy5lMKptfMIlby/view?usp=sharing)

### Steps

1. Download and unzip the provided `data.zip` file.
2. Download the **MFM_pretrained_vit_base** pretrained weights.
3. Upload the extracted data folder **and** the pretrained weight file to **Google Drive**.
4. Open `main.ipynb` in **Google Colab** and run all cells sequentially.

> ⚠️ Make sure that Colab is set to use a **GPU runtime** (`Runtime → Change runtime type → GPU`).

---