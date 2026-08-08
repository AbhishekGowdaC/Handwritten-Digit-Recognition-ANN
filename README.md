# Handwritten Digit Recognition using ANN

A handwritten digit classification project built using **PyTorch** and the **MNIST dataset**. The project uses an Artificial Neural Network (ANN) to classify handwritten digits from **0 to 9**.

## 🚀 Project Overview

This project demonstrates how to build, train, validate, and test a fully connected Artificial Neural Network using PyTorch.

The model includes:

* Fully connected layers
* Batch Normalization
* ReLU activation
* Dropout regularization
* Adam optimizer
* Learning-rate scheduling
* Early stopping
* Best-model checkpointing

The trained model achieved **98.34% accuracy on the MNIST test dataset**.

## 🧠 Model Architecture

```text
Input Image
28 × 28 pixels
      ↓
Flatten
784 features
      ↓
Linear
784 → 256
      ↓
Batch Normalization
      ↓
ReLU
      ↓
Dropout (0.5)
      ↓
Linear
256 → 128
      ↓
Batch Normalization
      ↓
ReLU
      ↓
Dropout (0.5)
      ↓
Linear
128 → 10
      ↓
Digit Prediction
```

The 10 output classes represent:

```text
0  1  2  3  4  5  6  7  8  9
```

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

| Dataset  | Images |
| -------- | -----: |
| Training | 60,000 |
| Testing  | 10,000 |

The 60,000 training images are split into:

* **80% Training**
* **20% Validation**

Each image has a size of **28 × 28 pixels**.

## ⚙️ Training Configuration

| Parameter      | Value             |
| -------------- | ----------------- |
| Framework      | PyTorch           |
| Batch Size     | 64                |
| Optimizer      | Adam              |
| Learning Rate  | 0.001             |
| Weight Decay   | 1e-4              |
| Loss Function  | CrossEntropyLoss  |
| Scheduler      | ReduceLROnPlateau |
| Early Stopping | Enabled           |
| Maximum Epochs | 50                |
| Dropout        | 0.5               |

## 📈 Results

The model achieved:

* **Best Validation Accuracy:** 98.22%
* **Test Accuracy:** **98.34%**

The model uses early stopping to prevent unnecessary training once the validation loss stops improving.

## 📁 Project Structure

```text
Handwritten-Digit-Recognition-ANN/
│
├── model.py
├── train.py
├── test.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
├── digit.png
│
├── checkpoints/
│   └── best_model.pth
│
├── model/
│   └── mnist_model.keras
│
└── data/
```

> `data/` and the virtual environment should not be uploaded to GitHub.

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/AbhishekGowdaC/Handwritten-Digit-Recognition-ANN.git
cd Handwritten-Digit-Recognition-ANN
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

Run:

```bash
python train.py
```

The training script:

1. Downloads the MNIST dataset.
2. Splits the training data into training and validation sets.
3. Creates the ANN.
4. Trains the model.
5. Validates the model after every epoch.
6. Adjusts the learning rate using `ReduceLROnPlateau`.
7. Saves the best model.
8. Uses early stopping when validation loss stops improving.

The best model is saved as:

```text
checkpoints/best_model.pth
```

## 🧪 Test the Model

Run:

```bash
python test.py
```

The model is evaluated on the 10,000 unseen MNIST test images.

Expected result:

```text
Test Accuracy: 98.34%
```

## ✍️ Predict a Handwritten Digit

The project also supports prediction from an individual image.

Place an image named:

```text
digit.png
```

in the project directory and run:

```bash
python predict.py
```

The image is:

1. Converted to grayscale.
2. Resized to 28 × 28.
3. Converted to a PyTorch tensor.
4. Passed through the trained ANN.
5. Classified into one of the ten digit classes.

Example:

```text
========== PREDICTION ==========
Predicted Digit: 7
================================
```

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pillow
* MNIST

## 🎯 Key Learning Outcomes

This project demonstrates:

* Building an ANN using PyTorch
* Working with the MNIST dataset
* DataLoader and batch processing
* Training and validation loops
* Forward propagation
* Backpropagation
* Cross-entropy loss
* Adam optimization
* Batch normalization
* Dropout regularization
* Learning-rate scheduling
* Early stopping
* Model checkpointing
* Model evaluation
* Image-based inference

## 👨‍💻 Author

**Abhishek Gowda C**

GitHub: [AbhishekGowdaC](https://github.com/AbhishekGowdaC)
