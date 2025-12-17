# Generative Model Classifier

본 프로젝트는 다양한 생성 모델(Generative Model)에 의해 생성된 이미지를 분류하는 실험을 포함합니다. CLIP, DINO, MFM과 같은 여러 특징 추출기를 사용하며, 표준적인 Fine-tuning 방식과 Deep Prompt-Tuning 방식을 모두 지원합니다.

## 설치

실험을 실행하기 전에, 필요한 라이브러리를 설치해야 합니다. 주요 의존성은 `MFM/requirements.txt`에 명시되어 있습니다.

```bash
# MFM 및 기타 주요 라이브러리 설치
pip install -r MFM/requirements.txt
```
*참고: `transformers`, `torch`, `tqdm`, `scikit-learn`과 같은 라이브러리도 필요합니다.*

## 실험 실행 방법

모든 실험은 Python 스크립트와 `configs/` 디렉토리에 위치한 JSON 설정 파일을 통해 관리됩니다.

### 1. Fine-tuning 실험

표준 Fine-tuning 실험(CLIP, DINO, MFM-CLIP)은 통합된 `main.py` 스크립트를 사용하여 실행합니다. `--config` 인자에 원하는 실험의 설정 파일 경로를 전달하여 실행할 수 있습니다.

**기본 사용법:**
```bash
python main.py --config configs/<config_file>.json
```

**실행 예시:**

*   **CLIP Fine-tuning:**
    ```bash
    python main.py --config configs/clip.json
    ```

*   **DINO Fine-tuning:**
    ```bash
    python main.py --config configs/dino.json
    ```

*   **MFM + CLIP Fine-tuning:**
    ```bash
    python main.py --config configs/mfm_clip.json
    ```

### 2. Deep Prompt-Tuning 실험

Deep Prompt-Tuning 실험은 구조의 특수성으로 인해 별도의 전용 스크립트로 실행됩니다. 각 스크립트의 설정 파일은 마찬가지로 `configs/` 디렉토리에 있습니다.

**기본 사용법:**
```bash
python <script_name>.py --config configs/<config_file>.json
```

**실행 예시:**

*   **CLIP + Deep Prompt-Tuning:**
    *   `clip_prompt.py` 스크립트를 사용합니다.
    ```bash
    python clip_prompt.py --config configs/clip_prompt.json
    ```

*   **MFM + CLIP + Deep Prompt-Tuning:**
    *   `mfm_clip_prompt.py` 스크립트를 사용합니다.
    ```bash
    python mfm_clip_prompt.py --config configs/mfm_clip_prompt.json
    ```
