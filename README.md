# Multi-Class Neuronal Cell Classification in Histological Images from HI-Impacted Fetal Sheep

## Project Overview

This repository contains the implementation for the research paper: **Multi-class Neuronal Cell Classification in Histological Images from HI-Impacted Fetal Sheep using Deep Neural Networks**. The project automates the classification of neuronal cells in fetal sheep brain tissue affected by Hypoxic-Ischemic Encephalopathy (HIE). Cells are classified into three categories: Normal, Intermediate, and Pyknotic, based on damage severity.

The dataset consists of 4,008 histological images from six brain regions (CA1, CA3, CA4, Dentate Gyrus, PS1, PS2) across sham, ischemia, and ischemia+hypothermia groups. Nine deep learning models were fine-tuned using 5-fold cross-validation, including CNNs, Vision Transformers (ViTs), and hybrids. Key results include:
- Top performers: Panda-ViT (87.6% accuracy), ViT-384 (87.0%), Hybrid-ViT (86.7%).
- EfficientNet-b7 achieved the highest ROC AUC (0.872) for the Intermediate class on external validation.
- The Hybrid-ViT was selected as optimal for preclinical HIE research due to pyknotic cell sensitivity, regional consistency, and moderate inference time.
- Performance saturation observed, likely due to dataset limitations rather than architecture.

This work extends a prior dataset of 1,500 cells and benchmarks models for an automated neuron segmentation/classification pipeline, accelerating HIE drug development.

The code handles data loading, model initialization (e.g., ViT, Swin, EfficientNet), training with oversampling and augmentations, evaluation (metrics like ROC AUC, confusion matrices), and visualizations (loss/accuracy plots, Grad-CAM heatmaps, ROC curves).

## Key Competencies and Technologies

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-green?logo=huggingface)
![Timm](https://img.shields.io/badge/Timm-0.9+-yellow)
![Albumentations](https://img.shields.io/badge/Albumentations-1.3+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-brightgreen?logo=scikit-learn)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.10+-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-red?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-yellow?logo=pandas) (via dependencies)
![Pillow](https://img.shields.io/badge/Pillow-9.4+-lightgrey?logo=python)
![Tqdm](https://img.shields.io/badge/Tqdm-4.65+-blue)
![PyTorch-Grad-CAM](https://img.shields.io/badge/PyTorch--Grad--CAM-1.4+-red)
![ImageHash](https://img.shields.io/badge/ImageHash-4.3+-green)
![ImageIO](https://img.shields.io/badge/ImageIO-2.27+-yellow)
![PSUtil](https://img.shields.io/badge/PSUtil-5.9+-lightgrey)
![Hashlib](https://img.shields.io/badge/Hashlib-Standard-blue?logo=python)
![Multiprocessing](https://img.shields.io/badge/Multiprocessing-Standard-orange?logo=python)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Models-brightgreen)
![Vision Transformers](https://img.shields.io/badge/Vision%20Transformers-ViT/Swin/Hybrid-yellow)
![CNNs](https://img.shields.io/badge/CNNs-EfficientNet/ResNet/Custom-red)
![Oversampling](https://img.shields.io/badge/Oversampling-SMOTE/ADASYN/BorderlineSMOTE-lightgrey)
![Augmentations](https://img.shields.io/badge/Augmentations-Albumentations-blue)
![Cross-Validation](https://img.shields.io/badge/Cross--Validation-5--Fold-green)
![Metrics](https://img.shields.io/badge/Metrics-ROC%20AUC/Confusion%20Matrix/Classification%20Report-orange)
![Visualization](https://img.shields.io/badge/Visualization-GradCAM/Plots/Heatmaps-yellow)
![Research](https://img.shields.io/badge/Research-Project-red)
![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-Histology-brightgreen)
![High-Performance Computing](https://img.shields.io/badge/HPC-NeSI%20Cluster-lightgrey) (via NeSI paths in code)

### Languages and Core Competencies
- **Python**: Primary language for all scripting, model training, and evaluation. Competencies include data processing, neural network implementation, multiprocessing for parallel training, and result aggregation/visualization.
- **Deep Learning Frameworks**: PyTorch for core model building and training; Hugging Face Transformers for ViT/Swin/Hybrid models; Timm for EfficientNet/SwinV2.
- **Data Handling**: NumPy/SciPy for numerical operations; Pillow/OpenCV/ImageIO for image loading/processing; Albumentations for augmentations; Imbalanced-Learn for handling class imbalance via SMOTE variants.
- **Machine Learning Tools**: Scikit-Learn for metrics (ROC AUC, confusion matrices, classification reports), StratifiedKFold for cross-validation.
- **Visualization and Interpretability**: Matplotlib for plotting (loss/accuracy curves, ROC, boxplots); PyTorch-Grad-CAM for model explainability via heatmaps.
- **Utilities**: Tqdm for progress tracking; PSUtil/GC for memory management; Hashlib/ImageHash for data integrity; Multiprocessing for GPU-parallel model training.
- **Domain Skills**: Medical image classification, handling imbalanced datasets, fine-tuning pre-trained models (ImageNet/Histology-specific), performance benchmarking across architectures.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/multi-class-neuronal-classification.git
   cd multi-class-neuronal-classification
   ```

2. Install dependencies (Python 3.12 recommended):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers timm albumentations opencv-python numpy scipy matplotlib pandas pillow tqdm imbalanced-learn pytorch-grad-cam imageio imagehash psutil
   ```

   Note: Ensure CUDA is installed for GPU acceleration (code uses `torch.cuda` for multi-GPU training).

3. Download the dataset (not included due to size/sensitivity; paths in code reference `/home/ibou703/00_nesi_projects/uoa00414/Ines/Training2/Oversampling&ViT/`).

## Usage

1. **Data Preparation**: Run the script to load and preprocess images/masks from specified directories. It renames files and loads RGB images with masks.

2. **Training**:
   ```
   python clean_training_code.py
   ```
   - Trains models in batches using available GPUs.
   - Supports models: EfficientNet, ViT-patch32-384, ViT-patch32-224, Swin, Swin-v2, ResNet-50, Hybrid-ViT-384, Panda-ViT, CustomNet, Hybrid2.
   - Uses oversampling (SMOTE/ADASYN/BorderlineSMOTE), augmentations, and 5-fold cross-validation.
   - Saves results to `/Results/` (JSON files per model, aggregated results).

3. **Evaluation and Visualization**:
   - Generates plots: Loss/accuracy curves, ROC per fold/class, confusion matrices, Grad-CAM heatmaps.
   - Aggregates results into `aggregated_results.json`.
   - Boxplots/scatter plots for model comparison.

4. **Customization**: Modify `model_names` list or hyperparameters (e.g., epochs=50, batch_size=16, lr=1e-5) in the script.

## Results and Visualizations

- Output directories: `/Accuracy_and_Loss/`, `/ROC_curves/`, `/Confusion_matrices/`, `/GradCAM/`.
- Key files: `aggregated_results.json` (all metrics), visualization PNGs (e.g., `5_best_validation_accuracy_boxplot.png`).
- Example: Training/validation accuracy/loss plots per model/fold/class, ROC AUC distributions.

## Contributing

This is a research project. For collaborations or extensions, contact via GitHub issues.

## License

Creative Commons Attribution (CC BY) license. See the paper for full citation.

## Acknowledgments

- Data from University of Auckland, Department of Physiology.
- Built on NeSI HPC cluster for training.
- Paper PDF included in repo for reference.
