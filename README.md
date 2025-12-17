# Experiment Execution Guide

This project was designed to be executed in a **GPU-enabled environment**. To ensure reproducibility and ease of setup, **all experiments were conducted using Google Colab via `main.ipynb`**.

---

## 🚀 Recommended: Run on Google Colab (GPU)

### Steps

1. Unzip the provided `data.zip` file.
2. Upload the extracted data folder to **Google Drive**.
3. Open [`main.ipynb`](main.ipynb) in **Google Colab** and run all cells sequentially.

> ⚠️ Make sure that Colab is set to use a **GPU runtime** (`Runtime → Change runtime type → GPU`).

---

## 💻 Alternative: Run Locally

If you prefer to run the experiments in a local environment, follow the steps below.

### Setup

1. Unzip the provided `data.zip` file.
2. Place the extracted data directory into `./data`.

---

## 🧪 Experiment Scripts

### 1. Linear Probing

- **CLIP**
  ```bash
  python main.py --config configs/clip.json
  ```

- **DINO**
  ```bash
  python main.py --config configs/dino.json
  ```

- **MFM + CLIP**
  ```bash
  python main.py --config configs/mfm_clip.json
  ```

---

### 2. With Deep Prompt Tuning

- **CLIP + Deep Prompt Tuning**
  ```bash
  python clip_prompt.py --config configs/clip_prompt.json
  ```

- **MFM + CLIP + Deep Prompt Tuning**
  ```bash
  python mfm_clip_prompt.py --config configs/mfm_clip_prompt.json
  ```

---
