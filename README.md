# Tire Texture Classifier

This project involves training a neural network model to classify tire textures as either "Normal Tire" or "Cracked Tire." The model is built using the Keras library and is trained on a dataset of tire texture images. The trained model is then incorporated into a simple GUI application using Tkinter for image classification.

## Files Overview

### 1. **model.ipynb**
   - Jupyter Notebook file containing the code for data analysis, visualization, neural network model creation, training, and evaluation.
   - Utilizes the Keras library for building a Convolutional Neural Network (CNN).
   - Implements data augmentation techniques for training dataset improvement.
   - Saves the trained model as 'model3.hdf5'.
   - Evaluates the model's performance on the testing dataset.

### 2. **app.py**
   - Python script for a GUI application using Tkinter.
   - Loads the trained model ('model3.hdf5') and defines classes for classification.
   - Allows users to upload an image for classification using the trained model.
   - Displays the selected image and the predicted class (Normal Tire or Cracked Tire).


## Running the Application

1. **Install Dependencies**
   pip install requirements.txt
