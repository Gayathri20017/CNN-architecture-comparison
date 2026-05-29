# CNN Architecture Comparison

A machine learning project comparing different CNN (Convolutional Neural Network) architectures for plant disease classification. This project evaluates **VGG16** and **ResNet50** models on their performance in classifying different types of plant leaf diseases.

## Project Overview

The project implements transfer learning techniques to classify plant leaf diseases into three categories:
- Algal Leaf
- Brown Blight
- White Spot

## Dataset

- **Total Images:** 368 images
- **Classes:** 3 disease types
- **Split:** 80% training, 20% validation (stratified)
- **Image Size:** 224×224 pixels
- **Data Format:** ImageFolder structure

## Technologies & Libraries

- **Framework:** PyTorch & TorchVision
- **Hardware:** GPU (CUDA-enabled recommended)
- **Key Libraries:**
  - `torch` - Deep learning framework
  - `torchvision` - Computer vision utilities
  - `scikit-learn` - Model evaluation metrics
  - `matplotlib` & `seaborn` - Visualization
  - `numpy` - Numerical computations

## Project Structure

```
CNN architecture comparison/
├── CNN architecture comparison_PythonScript.ipynb  # Main Jupyter notebook
├── CNN architecture comparison_report.pdf          # Detailed analysis report
└── README.md                                       # This file
```

## Model Architectures

### VGG16
- **Pre-trained on:** ImageNet
- **Strategy:** Fine-tuning with partial unfreezing
  - Frozen layers: All feature extraction layers except the last 2
  - Trainable layers: Last convolutional block + classifier
- **Output:** 3 neurons (for 3 classes)

### ResNet50
- **Pre-trained on:** ImageNet
- **Strategy:** Fine-tuning with selective unfreezing
  - Frozen layers: ResNet blocks 1-3 + BatchNorm layers
  - Trainable layers: Layer4 (final residual block) + fully connected layer
- **Output:** 3 neurons (for 3 classes)

## Data Processing

### Augmentation (Training Data)
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness: 0.2, contrast: 0.2)
- Resize to 224×224
- Normalization with ImageNet statistics

### Validation Data
- Resize to 224×224
- Normalization with ImageNet statistics
- No augmentation applied

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Device | GPU (CUDA) / CPU (fallback) |

## Key Features

✅ **Transfer Learning:** Pre-trained models from ImageNet  
✅ **Data Augmentation:** Improves model generalization  
✅ **Stratified Split:** Balanced train-test split  
✅ **Model Checkpointing:** Saves best model weights  
✅ **Comprehensive Metrics:** Accuracy, loss, confusion matrices  
✅ **Learning Rate Scheduling:** Dynamic learning rate adjustment  
✅ **Evaluation Metrics:** Classification reports with precision, recall, F1-score  

## Usage

### Prerequisites
```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

### Running the Project
1. Open `CNN architecture comparison_PythonScript.ipynb` in Jupyter Notebook or Google Colab
2. Ensure GPU acceleration is enabled for faster training
3. Update the data path to point to your dataset location
4. Run all cells sequentially

### Expected Output
- Training progress for each epoch
- Best model weights saved as `.pth` files
- Accuracy and loss comparisons between models
- Confusion matrices and classification reports
- Performance visualizations

## Results & Analysis

The project evaluates both architectures based on:
- **Training Accuracy** - How well models learn the training data
- **Validation Accuracy** - Generalization performance
- **Loss Curves** - Convergence behavior
- **Confusion Matrices** - Misclassification patterns
- **Classification Reports** - Precision, recall, F1-scores per class

Detailed analysis and results are available in the accompanying PDF report.

## Environment

- **Platform:** Google Colab / Local GPU Environment
- **Python Version:** Python 3.x
- **GPU:** NVIDIA GPU with CUDA support (recommended for faster training)

## Notes

- Models are initialized with ImageNet pre-trained weights
- Fine-tuning approach balances transfer learning benefits with task-specific adaptation
- Stratified splitting ensures balanced class distribution in train/val sets
- Best model weights are automatically saved during training

---

For detailed methodology, results, and analysis, please refer to `CNN architecture comparison_report.pdf`.
