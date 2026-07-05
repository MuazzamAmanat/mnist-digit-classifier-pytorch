A deep learning project that classifies handwritten digits (0–9) using a Fully Connected Neural Network (FCNN) built with PyTorch and trained on the MNIST dataset.

The project covers the full deep learning workflow — data loading and preprocessing, model training, evaluation, prediction, and model saving.


📌 Project Overview

The goal of this project is to train a neural network that recognizes handwritten digits with high accuracy, using the MNIST dataset of grayscale digit images.

This project covers:


Dataset loading
Data preprocessing
Neural network implementation
Model training
Model evaluation
Prediction on a sample test image
Model saving



🚀 Features

✔ PyTorch implementation
✔ MNIST dataset (auto-downloaded via torchvision)
✔ Automatic GPU/CPU detection
✔ Mini-batch training using DataLoader
✔ Feed-forward neural network
✔ Adam optimizer
✔ Cross entropy loss
✔ Test-set evaluation
✔ Saves trained model weights (.pth)
✔ Visualizes a sample prediction with matplotlib


🧠 Model Architecture

Input Image (28 × 28)
        ↓
    Flatten → 784 features
        ↓
  Linear(784 → 128)
        ↓
       ReLU
        ↓
  Linear(128 → 64)
        ↓
       ReLU
        ↓
  Linear(64 → 10)
        ↓
  Predicted Digit (0–9)


📂 Dataset

Dataset: MNIST (downloaded automatically by torchvision.datasets.MNIST)

PropertyValueTraining Images60,000Testing Images10,000Classes10Image Size28 × 28Channels1 (Grayscale)


⚙️ Training Configuration

ParameterValueOptimizerAdamLearning Rate0.001Loss FunctionCrossEntropyLossBatch Size64Epochs5


📊 Results

MetricValueTest Accuracy~97% (typical for this architecture)


Actual accuracy will vary slightly each run due to random weight initialization and shuffling — check your own console output after training.
