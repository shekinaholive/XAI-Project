# Pneumonia Detection and Explainable AI (XAI) Project

This repository contains the code and resources for a project focused on pneumonia detection using chest X-ray images and Explainable AI (XAI) techniques. The project employs multiple machine learning models and XAI methods to provide interpretable and reliable predictions.

## Overview

The goal of this project is to develop and evaluate pneumonia detection models and explain their predictions using state-of-the-art XAI techniques. This is achieved by:

1. Training and evaluating models like:
    * Custom CNN (from scratch)
    * Pretrained ResNet-18
    * Pretrained VGG16

2. Applying XAI techniques, including:
    * Grad-CAM (Gradient-weighted Class Activation Mapping)
    * Occlusion Sensitivity
3. Building an interactive Gradio-based dashboard to visualize predictions and explanations.

## Dataset
* Source: Chest X-Ray Images (Pneumonia)

## Models and Techniques
### Models
1. Custom CNN: A basic convolutional neural network trained from scratch for binary classification.
2. ResNet-18: A pretrained deep learning model fine-tuned for pneumonia detection.
3. VGG16: Another pretrained model adapted for the same task.

### XAI Methods
1. Grad-CAM: Highlights the regions in the image that most influence the model’s prediction.
2. Occlusion Sensitivity: Analyzes how removing parts of the image affects the model’s confidence.

## Dependencies
* Python 3.7+
* PyTorch
* Torchvision
* Gradio
* Captum
* Additional libraries: NumPy, Matplotlib, Seaborn, Pandas, PIL

