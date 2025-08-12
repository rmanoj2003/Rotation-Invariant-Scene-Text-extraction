# Rotation-Invariant-Scene-Text-extraction

This project presents a deep learning-based approach for robust scene text detection and recognition, specifically designed to handle rotated, curved, distorted, and complex-background text in real-world images. Unlike traditional OCR systems, this model employs a Vision Transformer (ViT)-based architecture with a specialized detection pipeline to achieve rotation invariance, improved generalization, and higher accuracy across diverse datasets.

The system operates in two main phases:

Text Detection – Locates text regions using a ResNet50 backbone, Feature Pyramid Network (FPN), and specialized shrink-mask and reinforcement-offset mapping.

Text Recognition – Uses a Vision Transformer to predict character sequences without requiring expensive character-level annotations.

Key Features
Rotation-invariant text extraction for real-world images

Handles multi-oriented, curved, and low-resolution text

Robust to background clutter, occlusions, and noise

Parallelized character recognition for faster inference

Eliminates the need for detailed character-level annotations

Adaptable to multiple datasets and scenarios (street signs, advertisements, container numbers, etc.)

Technical Details
Language: Python

Frameworks: PyTorch / TensorFlow (depending on implementation)

Core Components:

Text Detection Module:

Backbone: ResNet50 for multi-scale feature extraction

FPN for merging low- and high-level features

Shrink-mask & reinforcement-offset map generation

Super Pixel Window (SPW) for improved boundary accuracy

Text Recognition Module:

Vision Transformer (ViT) for parallel character sequence prediction

CTC decoder for handling variable-length outputs

Loss Functions:

Dice loss for segmentation (shrink-mask)

Ratio loss for offset & SPW maps

Cross-entropy for character prediction

Datasets Used: SynthText, ICDAR2015, SCUT-CTW1500, CUTE80
