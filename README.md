# skin-disease-Class

This project implements a skin disease detection model using a Convolutional Neural Network (CNN) based on the ResNet-50 architecture. The model is trained to classify images of skin diseases into eight categories, utilizing transfer learning with a pre-trained model. The model achieved an accuracy of **97%** on the validation dataset.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)

## Features

- Image classification for eight types of skin diseases.
- Utilizes transfer learning with ResNet-50.
- Performance evaluation using confusion matrix and classification report.
- Visualizations of training and validation loss and accuracy.
- Achieved 97% accuracy on the validation dataset.

## Technologies Used

- Python
- PyTorch
- torchvision
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Dataset

The dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset) and contains images of skin diseases, organized into training and validation sets.

## Installation

To run this project, you need to have the following installed:

1. Python 3.x
2. Required libraries can be installed using pip:

   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib seaborn

## Training the Model
The model is trained using the train_model function defined in the code. The training process includes:

    Data augmentation for the training set.
    Loss calculation using Cross Entropy Loss.
    Optimization using the Adam optimizer.
    Saving the best model based on validation accuracy.

## Evaluation
The evaluate_model function evaluates the trained model on the validation dataset. It outputs:

    Confusion matrix
    Classification report
    Visualization of the confusion matrix
    
