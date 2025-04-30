# Jellyfish Image Classification Using Deep Learning Models

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset and Preprocessing](#dataset-and-preprocessing)
5. [Model Architectures](#model-architectures)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Visualization](#visualization)
10. [Future Work](#future-work)

## Introduction
This project explores automated classification of jellyfish species from underwater images using four deep learning architectures:
- **Custom CNN:** A compact network with three convolutional blocks suited for limited compute.
- **ResNet-50:** Leverages residual connections in a 50-layer network via transfer learning.
- **DenseNet-121:** Utilizes dense blocks for feature reuse and parameter efficiency.
- **Vision Transformer (ViT-B/16):** Applies self-attention on 16×16 image patches for state-of-the-art performance.

Accurate identification aids marine biologists and conservationists in monitoring species distribution and population health.

## Project Structure
```bash
├── data/
│   ├── raw/                  # Original images organized by class
│   └── processed/            # Augmented and resized images
├── notebooks/                # Jupyter notebooks for EDA and analysis
├── src/
│   ├── data_preprocessing.py # Data loading and augmentation
│   ├── models/               # Model definitions (CNN, ResNet, DenseNet, ViT)
│   ├── train.py              # Training script with CLI options
│   ├── evaluate.py           # Model evaluation and metrics
│   └── visualize.py          # Plots for loss, accuracy, confusion matrices
├── checkpoints/              # Saved model weights and logs
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/jellyfish-classification.git
   cd jellyfish-classification
   ```
2. **Set up Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Optional:** Install GPU support
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
   ```

## Dataset and Preprocessing
- **Source:** Download from [Kaggle Jellyfish Images](https://www.kaggle.com/) or NOAA public archives.
- **Classes:** Moon Jellyfish, Box Jellyfish, Lion’s Mane, Sea Nettle, Blue Blubber.

### Preprocessing Steps
Run:
```bash
python src/data_preprocessing.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --img_size 224 \
    --augmentation "flip,rotate,zoom,brightness" \
    --split_ratio 0.8
```
Processed data structure:
```
data/processed/train/{class_name}/
data/processed/val/{class_name}/
```

## Model Architectures
- **Custom CNN:** Three Conv→ReLU→MaxPool blocks + two dense layers.
- **ResNet-50:** Initialized with ImageNet weights; last block fine-tuned.
- **DenseNet-121:** Retains pretrained features; adds a custom classification head.
- **ViT-B/16:** Patch embedding size 16; fine-tuned on jellyfish data.

Details available in [`notebooks/Model_Stats.ipynb`](notebooks/Model_Stats.ipynb).

## Training
```bash
python src/train.py \
    --model resnet50 \
    --data_dir data/processed \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --optimizer adam \
    --checkpoint_dir checkpoints/resnet50
```
- **optimizer:** Options: `sgd`, `adam`, `adamw`.
- **learning_rate:** Starting LR; scheduler reduces LR on plateau.
- **checkpoint_dir:** Save best weights.

TensorBoard logs:
```bash
tensorboard --logdir runs/
```

## Evaluation
```bash
python src/evaluate.py \
    --model resnet50 \
    --weights checkpoints/resnet50/best.pth \
    --data_dir data/processed/val
```
Metrics:
- Accuracy
- Precision & Recall (per class)
- F1-score
- Confusion Matrix

Reports saved as CSV in `runs/evaluation/`.

## Results
| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|---------:|----------:|-------:|---------:|
| Custom CNN    | 84.2%    | 85.1%     | 83.5%  | 84.3%    |
| ResNet-50     | 87.4%    | 88.2%     | 86.7%  | 87.4%    |
| DenseNet-121  | 89.1%    | 90.0%     | 88.3%  | 89.1%    |
| ViT-B/16      | 92.3%    | 93.1%     | 91.5%  | 92.3%    |

See `notebooks/Results_Analysis.ipynb` for per-class breakdown.

## Visualization
```bash
python src/visualize.py --model resnet50 --output_dir runs/visuals
```
Plots include:
- Loss & accuracy curves
- Per-class ROC
- Confusion matrices

## Future Work
- **Dataset Expansion:** More species & habitats.
- **Multimodal Fusion:** Image + metadata.
- **Edge Deployment:** Quantize & optimize for inference.
- **AutoML:** Neural architecture search.

---
*This README was generated for personal use.*