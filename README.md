# Facial Palsy Severity Assessment Using Deep Learning

A deep learning–based computer vision system for automated assessment of **facial palsy severity** from facial-region images. The project uses a transfer-learning approach with a ResNet-based convolutional neural network to classify facial palsy into four severity categories and provides visual interpretability through Grad-CAM.

> **Project Status:** Ongoing
> **Domain:** Healthcare AI / Medical Computer Vision
> **Task:** Multi-class Facial Palsy Severity Classification
> **Framework:** PyTorch
> **Deployment:** Web-based inference application


## Overview

Facial palsy is a neurological condition characterized by weakness or paralysis of the muscles responsible for facial movement. Clinical assessment commonly relies on visual examination of facial symmetry and movement.

This project explores the use of **deep learning and computer vision to assist in the automated assessment of facial palsy severity**.

Instead of treating the problem as a simple binary classification task, the system performs **four-class severity classification**:

* Mild
* Moderate
* Moderate Severe
* Severe

The model analyzes facial regions associated with asymmetry, including the:

* Eye region
* Eyebrow region
* Mouth region

The objective is to develop a reproducible AI pipeline that combines **data preprocessing, patient/sample-level splitting, transfer learning, model evaluation, and explainable AI**.


## Key Features

### Deep Learning Classification

A ResNet-based convolutional neural network is used with transfer learning to extract discriminative facial features.

The model is fine-tuned for four-class facial palsy severity classification.

### Region-Based Analysis

The dataset contains facial-region images corresponding to:

* Eyes
* Eyebrows
* Mouth

These regions contain clinically relevant information about facial asymmetry and muscle dysfunction.

### Four-Level Severity Classification

The system predicts one of four severity categories:

| Class           | Description                           |
| --------------- | ------------------------------------- |
| Mild            | Limited facial asymmetry              |
| Moderate        | Noticeable facial weakness            |
| Moderate Severe | Significant facial dysfunction        |
| Severe          | Pronounced facial paralysis/asymmetry |

### Explainable AI

Grad-CAM is incorporated to visualize the regions that contribute most strongly to the model's prediction.

This improves model interpretability and allows inspection of whether the network is focusing on meaningful facial regions rather than irrelevant image artifacts.

### Sample-Level Dataset Splitting

A major focus of the preprocessing pipeline was preventing data leakage.

Images originating from the same underlying sample/subject were grouped before splitting into training, validation, and testing sets.

This provides a more realistic estimate of model generalization than randomly splitting individual images.


# Dataset

The project uses an updated facial palsy dataset containing images from three facial regions.

After preprocessing and cleaning, the dataset contains:

**13,146 images**

### Region Distribution

| Facial Region |     Images |
| ------------- | ---------: |
| Eyebrow       |      6,275 |
| Eye           |      3,725 |
| Mouth         |      3,146 |
| **Total**     | **13,146** |

### Severity Distribution

| Severity        |     Images |
| --------------- | ---------: |
| Mild            |      1,590 |
| Moderate        |      4,924 |
| Moderate Severe |      5,446 |
| Severe          |      1,186 |
| **Total**       | **13,146** |

The dataset contains a natural class imbalance, with the Moderate Severe category representing the largest proportion of samples.

Rather than relying solely on accuracy, model evaluation therefore considers class-sensitive metrics such as **Macro F1-score**.


# Data Augmentation

To improve robustness and reduce sensitivity to variations in facial image acquisition, the preprocessing pipeline includes augmented samples.

The augmentation set contains:

* Rotation
* Cropping
* Noise injection
* Translation

Approximate augmentation distribution:

| Type       | Images |
| ---------- | -----: |
| Original   |  6,248 |
| Rotated    |  2,431 |
| Cropped    |  2,211 |
| Noisy      |  1,548 |
| Translated |    708 |

Augmentation was applied as part of the training data preparation process while maintaining separation between training, validation, and test samples.


# Dataset Split

The cleaned dataset was divided into:

| Split      |     Images |
| ---------- | ---------: |
| Training   |      8,322 |
| Validation |      2,105 |
| Testing    |      2,719 |
| **Total**  | **13,146** |

The dataset was split at the sample/group level rather than treating every image as an independent observation.

This was specifically designed to reduce the possibility of **data leakage**, particularly where multiple facial-region images or augmented variants may originate from the same underlying sample.


# Model Architecture

The classification model is based on **ResNet18 with transfer learning**.

### Architecture Strategy

1. Load a ResNet18 model pretrained on ImageNet.
2. Replace the original classification head.
3. Adapt the network for four facial palsy severity classes.
4. Freeze most of the pretrained backbone initially.
5. Fine-tune deeper convolutional layers.
6. Train the classification head for the target task.

Transfer learning allows the model to leverage general visual representations learned from large-scale image datasets while adapting higher-level features to the medical imaging task.

# Training Configuration

The training pipeline uses:

* **Framework:** PyTorch
* **Architecture:** ResNet18
* **Input resolution:** 224 × 224
* **Batch size:** 8
* **Maximum epochs:** 20
* **Optimizer:** AdamW
* **Backbone learning rate:** 1e-4
* **Classifier learning rate:** 5e-4
* **Weight decay:** 1e-4
* **Learning-rate scheduler:** Cosine Annealing
* **Early stopping:** Enabled
* **Random seed:** 42

The best model is selected using **validation Macro F1-score** rather than accuracy alone.

This is particularly important because the dataset contains class imbalance.

# Evaluation

The model is evaluated using multiple metrics rather than relying solely on accuracy.

### Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion Matrix

### Why Macro F1?

For a medical classification problem with imbalanced severity classes, overall accuracy can hide poor performance on minority classes.

Macro F1 calculates the F1-score independently for each class and then gives each class equal weight.

This makes it more informative when evaluating whether the model performs reasonably across all severity levels.


# Explainability with Grad-CAM

Medical AI systems require more than predictive performance.

The project therefore incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the image regions influencing the prediction.

Grad-CAM produces a heatmap indicating areas that contribute strongly to the predicted class.

This can be used to investigate whether the model is focusing on relevant facial structures such as:

* Eye region
* Eyebrow region
* Mouth region
* Areas exhibiting facial asymmetry

The visualization is intended as a model interpretability mechanism and **not as a replacement for clinical assessment**.


# Inference Pipeline

The inference workflow follows the general pipeline:

```text
Input Facial Image
        ↓
Image Preprocessing
        ↓
Resize / Normalization
        ↓
Trained ResNet Model
        ↓
Severity Prediction
        ↓
Confidence Score
        ↓
Grad-CAM Visualization
        ↓
Web-Based Result
```

The application provides an interface through which an image can be submitted for model inference.


# Web Application

A web-based interface was developed to make the trained model accessible through a user-friendly workflow.

The application is designed around a healthcare-oriented interface and provides:

* Image upload
* Model inference
* Predicted severity
* Confidence information
* Explainability visualization
* Result presentation

The deployment layer separates the user interface from the underlying deep learning inference pipeline.

# Project Architecture

```text
facial-palsy-ai/
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── model.py
│   ├── inference.py
│   └── ...
│
├── outputs/
│   ├── models/
│   └── visualizations/
│
├── dataset/
│   ├── train/
│   ├── validate/
│   └── test/
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** The public repository may contain a reduced project structure for portfolio demonstration. Core training and inference implementation is maintained separately.

# Technologies Used

### Machine Learning

* Python
* PyTorch
* Torchvision
* Scikit-learn

### Computer Vision

* OpenCV
* PIL
* Image preprocessing and augmentation
* Grad-CAM

### Backend / Application

* Flask
* REST-based inference workflow

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Web-based deployment environment
* GitHub
* Vercel / cloud deployment components

# Challenges Addressed

## 1. Class Imbalance

The dataset contains significantly different numbers of samples across severity levels.

The training and evaluation strategy therefore emphasizes Macro F1 rather than accuracy alone.

## 2. Data Leakage

Multiple images can originate from the same underlying facial sample.

A naive image-level random split can therefore cause highly similar samples to appear in both training and testing data.

The project uses sample/group-level separation to reduce this risk.

## 3. Limited Medical Dataset Size

Rather than training a deep CNN completely from scratch, transfer learning was used to leverage pretrained visual representations.

## 4. Model Interpretability

A prediction alone is insufficient for a healthcare-oriented AI system.

Grad-CAM was incorporated to provide visual insight into the model's decision process.

## 5. Deployment

The project extends beyond model training into an end-to-end application pipeline, including preprocessing, inference, result generation, and web presentation.

---

# Results

The final trained model achieved approximately **76% test accuracy** on the prepared dataset.

Performance should be interpreted together with class-wise precision, recall, F1-score, Macro F1-score, and the confusion matrix rather than accuracy alone.

> **Important:** The reported performance is based on the project's prepared dataset and evaluation methodology. It should not be interpreted as clinical validation or evidence of diagnostic performance in a real-world patient population.

# Clinical Disclaimer

This project is an **academic and research-oriented AI prototype**.

It is not intended to:

* Diagnose facial palsy
* Replace a physician or neurologist
* Determine treatment
* Make independent clinical decisions
* Serve as a certified medical device

Predictions should be treated as experimental model outputs and require appropriate clinical validation before any real-world medical use.

# Future Improvements

Potential future development includes:

* Larger and more diverse clinical datasets
* Patient-level external validation
* Multi-center validation
* Better handling of class imbalance
* Facial landmark-based features
* Temporal analysis using facial movement/video
* Ensemble models
* Calibration of confidence scores
* Robustness testing across demographic and acquisition variations
* Clinical evaluation with healthcare professionals
* Prospective validation

# Project Significance

This project demonstrates an end-to-end **Healthcare AI / Medical Computer Vision workflow**, extending beyond simple model training.

The work covers:

**Dataset Engineering → Leakage Prevention → Data Augmentation → Transfer Learning → Severity Classification → Model Evaluation → Explainable AI → Backend Inference → Web Application → Deployment**

The primary objective is to demonstrate how deep learning can be integrated into a healthcare-oriented computer vision pipeline while maintaining attention to **model evaluation, interpretability, and responsible AI considerations**.

# Author

**Prachiti Shivalkar**

B.Tech Artificial Intelligence
SNDT Women's University

Interested in:

* Healthcare AI
* Medical Computer Vision
* Deep Learning
* Explainable AI
* AI-assisted Clinical Systems

## License

Copyright © 2026 Prachiti Shivalkar. All rights reserved.

This repository is provided for portfolio and educational viewing purposes only.

The project and its outputs may not be used for clinical diagnosis or medical decision-making.

