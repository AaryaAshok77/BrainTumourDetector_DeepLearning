# Brain Tumour Detection Neural Network

## Basic Setup

### Importing Packages

- **sys**
- **numpy**
- **torch**
- **torch.utils.data.Dataset, DataLoader**
- **glob**
- **matplotlib.pyplot**
- **sklearn.metrics.confusion_matrix, accuracy_score**
- **cv2**
- **torch.nn, torch.nn.functional**
- **seaborn**
- **sklearn.model_selection.train_test_split**

### Reading the Images

- Utilizing OpenCV (cv2) to read and preprocess MRI images.
- Resize the images to 128x128 pixels.
- Normalize the pixel values to [0, 1].
- Splitting the dataset into tumorous and healthy images.

### Visualizing the MRI Image

- Utilizing matplotlib.pyplot to visualize a random sample of healthy and tumorous MRI images.

## Creating MRI Dataset Class

- Implementing a custom PyTorch Dataset class to handle the MRI dataset.
- Includes functionality for loading, preprocessing, and splitting the dataset into training and testing subsets.
- The dataset class normalizes the image data.

## Creating MRI Model using CNN

- Defining a Convolutional Neural Network (CNN) model using PyTorch.
- The CNN model consists of convolutional layers followed by fully connected layers.
- Utilizing the Tanh activation function for the hidden layers and the Sigmoid activation function for the output layer.
- The model is trained to classify MRI images as either tumorous or healthy.

## Training and Testing

- Splitting the dataset into training and testing data using DataLoader.
- Training the CNN model using the Adam optimizer with a binary cross-entropy loss function.
- Monitoring the training and testing loss for each epoch.
- Visualizing the training and testing loss over multiple epochs.
- Observing potential overfitting of the model to the training data.

## Conclusion

- The Brain Tumour Detection Neural Network utilizes a CNN architecture to classify MRI images as tumorous or healthy.
- The model is trained and tested using PyTorch, achieving a certain level of accuracy.
- Further optimization and regularization techniques may be required to address overfitting issues observed during training.