# 📘 OCR Pipeline — Full Setup & Execution Guide

This repository provides a complete OCR processing pipeline covering all stages from data preprocessing, rotation correction, text detection, text recognition, to final output formatting and deployment.  
The environment and execution flow are designed for reproducibility, compatibility, and streamlined deployment.

## 🛠️ Environment Setup

The entire project is built on **Python 3.6** to maintain compatibility with legacy OCR frameworks such as PaddleOCR, MobileNetV3, and VietOCR.

### Create Virtual Environment
```bash
python3.6 -m env mc_ocr_env
```

### Activate Environment
**Windows**
```bash
mc_ocr_env\Scripts\activate.bat
```

**Linux/macOS**
```bash
source mc_ocr_env/bin/activate
```

## 📦 Dependency Installation

Install all base dependencies:
```bash
pip install -r mc_ocr/requirements.txt
```

Each module also contains its own `requirements.txt` if you want finer control:
- rotation_corrector/requirements.txt
- text_detector/requirements.txt
- text_classifier/requirements.txt

## 🚀 Pipeline Execution Flow

The full OCR pipeline consists of 6 major phases.

### 1️⃣ Data Filtering & Augmentation

Filter upright and horizontal samples:
```bash
python mc_ocr/rotation_corrector/process_mc_ocr_data.py
```

Generate synthetic training samples:
```bash
python mc_ocr/rotation_corrector/data_process.py
```

### 2️⃣ Train Rotation Correction Model (MobileNetV3)

```bash
python mc_ocr/rotation_corrector/train_config.py --cfg experiments/mobilenetv3_filtered_public_train.yaml
```

### 3️⃣ Set PYTHONPATH (Recommended)

**Windows**
```bash
set PYTHONPATH=%CD%
```

**Linux/macOS**
```bash
export PYTHONPATH=$(pwd)
```

### 4️⃣ Inference Pipeline

#### Text Detection (PaddleOCR)
```bash
python mc_ocr/text_detector/PaddleOCR/tools/infer/predict_det.py
```

#### Rotation Correction (MobileNetV3)
```bash
python mc_ocr/rotation_corrector/inference.py
```

#### Text Recognition (VietOCR)
```bash
python mc_ocr/text_classifier/pred_ocr.py
```

#### Final Aggregation / Submit
```bash
python mc_ocr/submit/submit.py
```

### 5️⃣ Web Application Deployment (Streamlit)

```bash
streamlit run mc_ocr/deployment/design.py
```

## ⚙️ Compatibility Notes

- Python **3.6 is required**; later versions may break PaddleOCR or MobileNetV3.
- The project uses multiple frameworks (**PaddlePaddle**, **PyTorch**) ⇒ strict version pinning recommended.
- For portability, consider:
```bash
conda env export > env.yml
```

## 📄 License
Add your license information here.

## 🙌 Acknowledgements
Thanks to the authors of PaddleOCR, MobileNetV3, VietOCR, and other open-source contributors.
