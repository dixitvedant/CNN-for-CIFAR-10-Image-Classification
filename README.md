# CNN for CIFAR-10 Image Classification

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10 dataset**.

The model learns visual features from images and classifies them into **10 different object categories** such as airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

This project demonstrates the **complete deep learning pipeline** including:

- Data preprocessing
- Image normalization
- CNN architecture design
- Model training
- Loss optimization
- Model evaluation

---

# Dataset

The model uses the **CIFAR-10 dataset**, which contains:

- 60,000 color images
- Image size: **32 × 32**
- **10 classes**
- 50,000 training images
- 10,000 testing images

Classes:

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

# Data Preprocessing

The following transformations are applied:

1. Convert image → Tensor
2. Scale pixel values (0–255 → 0–1)
3. Normalize values (-1 to +1)

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
```

Normalization is applied for **each RGB channel**.

---

# CNN Architecture

The network consists of **three convolution blocks** followed by **fully connected layers**.

### Architecture

```
Input Image
3 × 32 × 32

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
```

---

# Model Details

| Component | Value |
|--------|--------|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 64 |
| Input Image Size | 32×32 |

---

# Training Process

The training pipeline includes:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Weight updates using Adam optimizer

Example training loop:

```python
optimizer.zero_grad()
output = model(images)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
```

---

# Results

Training completed for **10 epochs**.

Final model performance:

```
Test Accuracy: 75.56%
```

The CNN successfully learns hierarchical visual features such as edges, textures, and object shapes.

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Deep Learning
- Convolutional Neural Networks
- Image Classification

---

# Future Improvements

Possible improvements include:

- Data augmentation
- Batch normalization
- Dropout layers
- Deeper CNN architectures
- Transfer learning (ResNet / VGG)
