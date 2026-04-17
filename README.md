# CIFAR-10 Image Classification using CNN (PyTorch + Streamlit)
link:https://cnn-for-cifar-10-image-classification-sxwwrugojxwscbf4rmc5rp.streamlit.app

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10 dataset**.

The trained model is deployed using **Streamlit**, allowing users to upload an image and get real-time predictions from the deep learning model.

This project demonstrates a **complete Deep Learning pipeline**:

- Data preprocessing
- CNN model design
- Model training
- Model evaluation
- Model deployment using Streamlit
- Real-time image prediction

---

## Live Application

After deployment, the app allows users to:

1. Upload an image
2. Model processes the image
3. Predicts object class
4. Displays prediction with confidence score

---

## Dataset

The model is trained on the **CIFAR-10 dataset**, which contains:

- 60,000 color images
- Image size: 32 × 32 pixels
- 10 object classes
- 50,000 training images
- 10,000 testing images

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset Source:
https://www.cs.toronto.edu/~kriz/cifar.html

---

## CNN Architecture

The CNN model consists of three convolution blocks followed by fully connected layers.

### Architecture Flow

Input Image (3 × 32 × 32)

Conv2D (3 → 32)  
ReLU  
MaxPooling  

Conv2D (32 → 64)  
ReLU  
MaxPooling  

Conv2D (64 → 128)  
ReLU  
MaxPooling  

Flatten  

Fully Connected (2048 → 256)  
ReLU  

Output Layer (256 → 10)

---

## Model Details

| Component | Value |
|----------|------|
| Framework | PyTorch |
| Dataset | CIFAR-10 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 64 |
| Input Size | 32×32 |
| Output Classes | 10 |

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- Deep Learning
- Convolutional Neural Networks

---
