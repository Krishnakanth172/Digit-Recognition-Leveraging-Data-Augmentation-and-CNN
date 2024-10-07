Project Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. 
The dataset consists of grayscale images of digits (0-9) that have been normalized and reshaped for CNN input. 
The model is enhanced using data augmentation techniques to improve generalization and accuracy. 
The final model is evaluated on the test data and provides predictions on unseen samples.

Key Features
CNN Architecture: A sequential model with convolutional layers, max-pooling, dropout, and dense layers to classify the digits.
Data Augmentation: Applied using ImageDataGenerator to improve model generalization through transformations like rotation, zoom, width/height shifts.
Training & Validation: The model is trained on the MNIST training set and validated on the test set over 10 epochs.
Accuracy & Loss: Plots of training and validation accuracy and loss are generated to visualize model performance.
Prediction: The model predicts individual test samples, displaying the true and predicted labels.

Dataset
The project uses the MNIST dataset, which consists of:
60,000 training images
10,000 testing images
The images are grayscale and have a size of 28x28 pixels. Each image represents a digit (0-9).

Model Architecture
The CNN architecture consists of the following layers:
Conv2D layer with 32 filters, followed by a MaxPooling2D layer
Conv2D layer with 64 filters, followed by another MaxPooling2D layer
Conv2D layer with 64 filters
Flatten layer to convert the 2D output to 1D
Dense layer with 64 units and ReLU activation
Dropout layer with 0.5 dropout rate for regularization
Dense layer with 10 units (for 10 output classes) and softmax activation for classification.

Requirements
Python 3.x
TensorFlow 2.x
NumPy
Matplotlib

You can install the dependencies using:
pip install tensorflow numpy matplotlib

How to Run
Clone the repository:
git clone <repository_url>
cd <repository_directory>

Run the Python script:
python mnist_cnn.py

The script will:
Load and preprocess the MNIST dataset
Train the CNN model with data augmentation
Plot training/validation accuracy and loss
Display predictions on sample test images

Output
Accuracy: The model achieves a test accuracy of approximately 99.25%.
Sample Prediction: A sample test image along with its predicted label is displayed using Matplotlib.

Results Visualization
During training, a plot of the training and validation accuracy across epochs will be generated. 
After training, you will see a sample image from the test set with its true and predicted labels.
