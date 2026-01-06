# Multi-Class Neuronal Cell Classification in Histological Images from HI-Impacted Fetal Sheep

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow?logo=huggingface)

![Timm](https://img.shields.io/badge/Timm-0.9+-red)
![Albumentations](https://img.shields.io/badge/Albumentations-1.3+-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-red?logo=scikit-learn)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.10+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-red?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-red?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-red?logo=pandas)
![Pillow](https://img.shields.io/badge/Pillow-9.4+-red?logo=python)
![Tqdm](https://img.shields.io/badge/Tqdm-4.65+-red)
![ImageIO](https://img.shields.io/badge/ImageIO-2.27+-red)
![ImageHash](https://img.shields.io/badge/ImageHash-4.3+-red)
![PSUtil](https://img.shields.io/badge/PSUtil-5.9+-red)
![Hashlib](https://img.shields.io/badge/Hashlib-Standard-red?logo=python)
![Multiprocessing](https://img.shields.io/badge/Multiprocessing-Standard-red?logo=python)

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Models-darkblue)
![Vision Transformers](https://img.shields.io/badge/Vision%20Transformers-ViT/Swin/Hybrid-darkblue)
![CNNs](https://img.shields.io/badge/CNNs-EfficientNet/ResNet/Custom-darkblue)
![Cross-Validation](https://img.shields.io/badge/Cross--Validation-5--Fold-darkblue)
![Metrics](https://img.shields.io/badge/Metrics-ROC%20AUC/Confusion%20Matrix/Classification%20Report-darkblue)
![High-Performance Computing](https://img.shields.io/badge/HPC-NeSI%20Cluster-darkblue)

![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-Histology-darkgreen)
![Research](https://img.shields.io/badge/Research-Article-black)


## Project Overview

This repository contains the code implementation for the research article: **Multi-class Neuronal Cell Classification in Histological Images from HI-Impacted Fetal Sheep using Deep Neural Networks**. The project focuses on automating the classification of neuronal cells in fetal sheep brain tissue affected by Hypoxic-Ischemic Encephalopathy (HIE), a condition leading to brain damage due to oxygen deprivation around birth. Cells are classified into three categories based on damage severity: Normal (uniform with clear nuclei), Intermediate (transitional fusion of nucleus and cytoplasm), and Pyknotic (severely damaged, small, circular, darkly stained with white halo).

The work extends a prior dataset from 1,500 to 5,763 manually segmented and classified cells across six brain regions (CA1, CA3, CA4, Dentate Gyrus, PS1, PS2) from sham, ischemia, and ischemia+hypothermia groups. An external validation set of 926 cells from MPA-treated models was used for generalization assessment.

Nine deep learning architectures were compared using 5-fold cross-validation: a custom CNN, ResNet-50, EfficientNet-b7, ViT-Base (224×224 and 384×384), Swin Transformer Large, Swin Transformer V2 Large, ViT-BiT Hybrid, and Panda-ViT (histology-pretrained). Models were adapted for 4-channel inputs (RGB + binary mask) to leverage spatial context for better discrimination.

Key achievements include:
- Demonstrating performance saturation likely due to dataset limitations, with global accuracies ranging 79.0-87.6% on external validation.
- Identifying Hybrid ViT as optimal for preclinical HIE research, balancing pyknotic sensitivity (66.2% recall), regional consistency (SD ±0.089), and moderate inference time (0.021s per cell).
- Showing transformer-based models excel in consistency across regions, while CNNs like EfficientNet-b7 achieve top intermediate class AUC (0.872).

This advances automated quantification for HIE drug development by improving accuracy over prior custom CNN (83.1%) and providing insights into architectural trade-offs.

### Languages and Core Competencies
- **Python**: Primary language for all scripting, model training, and evaluation. Competencies include data processing, neural network implementation, multiprocessing for parallel training, and result aggregation/visualization.
- **Deep Learning Frameworks**: PyTorch for core model building and training; Hugging Face Transformers for ViT/Swin/Hybrid models; Timm for EfficientNet/SwinV2.
- **Data Handling**: NumPy/SciPy for numerical operations; Pillow/OpenCV/ImageIO for image loading/processing; Albumentations for augmentations.
- **Machine Learning Tools**: Scikit-Learn for metrics (ROC AUC, confusion matrices, classification reports), StratifiedKFold for cross-validation.
- **Visualization and Interpretability**: Matplotlib for plotting (loss/accuracy curves, ROC, boxplots); brief use of PyTorch-Grad-CAM for heatmaps tests.
- **Utilities**: Tqdm for progress tracking; PSUtil/GC for memory management; Hashlib/ImageHash for data integrity; Multiprocessing for GPU-parallel model training.
- **Domain Skills**: Medical image classification, handling imbalanced datasets, fine-tuning pre-trained models (ImageNet/Histology-specific), performance evaluation across architectures.

## Dataset

The dataset comprises histological images from fetal sheep brains, sourced from the University of Auckland's Department of Physiology. It includes:
- Training set: 4,008 images expanded to 5,763 cells (1,791 Normal, 3,168 Intermediate, 804 Pyknotic).
- External validation: 926 cells (522 Normal, 339 Intermediate, 65 Pyknotic) from distinct MPA-treated models.
- Regions: Hippocampal (CA1, CA3, CA4, DG) and parasagittal cortical (PS1, PS2), capturing vulnerability gradients.
- Groups: Sham (baseline), Ischemia (untreated damage), Ischemia+Hypothermia (treatment effects).

Data preparation involved file renaming, loading RGB images with masks, and balancing via undersampling/oversampling to address class imbalance.

## Methodology

The code implements a comprehensive pipeline for model training and evaluation:
- **Data Loading and Preprocessing**: Loads images/masks from class-specific directories, renames files, and prepares datasets using PyTorch Datasets/DataLoaders. Applies transformations for 4-channel inputs.
- **Model Initialization**: Custom adaptations for 4-channel inputs and 3-class outputs:
  - CNNs: EfficientNet-b7, ResNet-50, CustomNet (skip connections, conv blocks).
  - ViTs: Large ViT (224/384), Panda-ViT (histology-pretrained).
  - Transformers: Swin Large, SwinV2 Large.
  - Hybrids: ViT-BiT Hybrid, Hybrid2 (EfficientNet + SwinV2).
- **Training**: 5-fold stratified cross-validation with early stopping. Uses Adam optimizer, cosine annealing scheduler, cross-entropy loss. Oversampling with SMOTE/ADASYN/BorderlineSMOTE variants; augmentations via Albumentations (geometric: flips, affine; photometric: blur, noise, contrast).
- **Evaluation**: Computes accuracy, loss (overall/per-class), ROC AUC (macro/per-class), confusion matrices. Generates Grad-CAM heatmaps for interpretability.
- **Visualization**: Plots training/validation curves (loss/accuracy overall/per-class), ROC curves, boxplots/scatter for model comparison, regional performance.
- **Parallel Processing**: Utilizes multiprocessing for multi-GPU training on NeSI HPC cluster, aggregating results in JSON.

<img width="834" height="713" alt="Capture d&#39;écran 2026-01-06 122741" src="https://github.com/user-attachments/assets/8a0d4479-469a-4b01-b9c9-0ef24490f978" />



## Results

Benchmarking revealed:
- **Top Performers**: Panda-ViT (87.6% ± 0.081 global accuracy), ViT-384 (87.0% ± 0.083), Hybrid-ViT (86.7% ± 0.089) on external validation.
- **Class-Specific**: High pyknotic AUC (0.845-0.980), healthy (0.887-0.928); intermediate hardest (0.837-0.872 AUC).
- **Consistency**: Transformers show lower regional variance (SD ±0.081-0.089) vs. CNNs (±0.101-0.177).
- **Inference**: Custom CNN fastest (0.002s/cell), Swin slowest (0.037s); Hybrid-ViT moderate (0.021s).
- **Saturation**: Narrow accuracy range (79-88%) indicates dataset limits over architecture.

Hybrid ViT optimal for HIE severity assessment due to superior pyknotic sensitivity and balance.

<img width="559" height="577" alt="Capture d&#39;écran 2026-01-06 122606" src="https://github.com/user-attachments/assets/ce90f7be-0da2-490f-bc50-a859e2326efd" />


<img width="914" height="463" alt="Capture d'écran 2026-01-06 123116" src="https://github.com/user-attachments/assets/f9353da8-8676-4004-80c2-7c272f9220b5" />

<img width="827" height="511" alt="Capture d'écran 2026-01-06 123121" src="https://github.com/user-attachments/assets/10080ec3-ecfa-46aa-9e52-a41f005065d3" />

<img width="870" height="487" alt="Capture d'écran 2026-01-06 122712" src="https://github.com/user-attachments/assets/c8ed3499-c108-44e2-b6fa-56d58144360b" />





## Acknowledgments

- Data from University of Auckland, Department of Physiology.
- Built on NeSI HPC cluster for training.
- Internship report included in repo. Please contact me at i.boulefred@gmail.com for further information.
