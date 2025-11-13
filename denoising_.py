#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D

# Load dataset
data = np.load(r'C:\Users\felin\Downloads\Image classification\mnist_compressed.npz')
X_train, X_test = data['train_images'], data['test_images']
y_train, y_test = data['train_labels'], data['test_labels']

# Normalize the data and add noise
X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
noise_factor = 0.6
x_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
x_test_noisy = X_test + noise_factor * np.random.normal(size=X_test.shape)

# Clip noisy images to ensure they stay in the range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reshape images for the model
X_train = X_train.reshape(-1, 28, 56, 1)
X_test = X_test.reshape(-1, 28, 56, 1)
x_train_noisy = x_train_noisy.reshape(-1, 28, 56, 1)
x_test_noisy = x_test_noisy.reshape(-1, 28, 56, 1)

# Display original and noisy images
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].imshow(X_train[0].reshape(28, 56), cmap='gray')
axes[1].imshow(x_train_noisy[0].reshape(28, 56), cmap='gray')
plt.tight_layout()
plt.show()

# Build the autoencoder model
model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 56, 1)),
    MaxPool2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPool2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(1, 3, activation='sigmoid', padding='same')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_noisy, X_train, epochs=5, batch_size=256, validation_data=(x_test_noisy, X_test))

# Save the model
model.save('./denoise_2.keras')

# Denoise the test images
denoised_images = model.predict(x_test_noisy)

# Plot original, noisy, and denoised images
n = 5  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 56), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 56), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].reshape(28, 56), cmap='gray')
    plt.title("Denoised")
    plt.axis("off")

plt.show()


# In[ ]:


The provided code trains a denoising autoencoder on the MNIST dataset to remove noise from images. The key steps in the code involve loading the dataset, adding noise, defining and training a convolutional autoencoder, and finally denoising the test images. Here's a detailed breakdown of each part of the code:

### 1. **Importing Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
```

- **NumPy** (`np`): Used for array operations, including manipulating and normalizing the image data.
- **Matplotlib** (`plt`): Used for visualizing images (both noisy and denoised).
- **TensorFlow** (`tf`): A popular deep learning library, used to define and train the autoencoder.
- **Keras** layers (`Conv2D`, `MaxPool2D`, `UpSampling2D`): These layers are used to build the convolutional autoencoder model.

### 2. **Loading the Dataset**

```python
data = np.load(r'C:\Users\felin\Downloads\Image classification\mnist_compressed.npz')
X_train, X_test = data['train_images'], data['test_images']
y_train, y_test = data['train_labels'], data['test_labels']
```

- The dataset is loaded from a compressed `.npz` file containing the MNIST dataset images and labels.
  - `train_images`: Training images.
  - `test_images`: Test images.
  - `train_labels`: Labels for the training images.
  - `test_labels`: Labels for the test images.

### 3. **Data Normalization and Noise Addition**

```python
X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
noise_factor = 0.6
x_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
x_test_noisy = X_test + noise_factor * np.random.normal(size=X_test.shape)
```

- **Normalization**: The pixel values of the MNIST images are scaled from the range `[0, 255]` to `[0, 1]` by dividing by 255. This helps in speeding up training.
- **Noise Addition**: Gaussian noise is added to both the training and test images to simulate noisy data. The `noise_factor` controls the strength of the noise. This simulates a noisy environment, which the autoencoder will learn to denoise.

```python
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

- **Clipping**: Ensures that the noisy images remain within the valid range `[0, 1]`. Without clipping, some pixel values might fall outside this range due to noise.

### 4. **Reshaping the Data for the Model**

```python
X_train = X_train.reshape(-1, 28, 56, 1)
X_test = X_test.reshape(-1, 28, 56, 1)
x_train_noisy = x_train_noisy.reshape(-1, 28, 56, 1)
x_test_noisy = x_test_noisy.reshape(-1, 28, 56, 1)
```

- **Reshaping**: The images are reshaped to have the dimensions `(batch_size, height, width, channels)`. Since the MNIST images are 28x28 pixels and are converted to grayscale (`1` channel), the shape is `(28, 28, 1)` for each image.
- **28x56**: There's a change in the original size from 28x28 to 28x56. This may be to adapt the images for the autoencoder structure, though resizing might have been done earlier (or implied in the data).

### 5. **Displaying Original and Noisy Images**

```python
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].imshow(X_train[0].reshape(28, 56), cmap='gray')
axes[1].imshow(x_train_noisy[0].reshape(28, 56), cmap='gray')
plt.tight_layout()
plt.show()
```

- **Visualization**: This code plots the first image from the training set (both the original and the noisy version) for comparison.
- **`imshow`**: Displays images in grayscale.
- **`tight_layout`**: Adjusts the layout to prevent overlap.

### 6. **Building the Autoencoder Model**

```python
model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 56, 1)),
    MaxPool2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPool2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(1, 3, activation='sigmoid', padding='same')
])
```

- **Conv2D Layer**: Performs convolution, a key operation for extracting features from images.
  - `32`: Number of filters (or kernels).
  - `3`: Kernel size, i.e., the filter is 3x3.
  - `activation='relu'`: The activation function, Rectified Linear Unit (ReLU), introduces non-linearity.
  - `padding='same'`: Ensures the output feature maps have the same spatial dimensions as the input.
- **MaxPool2D Layer**: Reduces the spatial dimensions (height and width) by a factor of 2, helping in down-sampling and feature extraction.
- **UpSampling2D Layer**: Upsamples the feature map by a factor of 2, effectively increasing the spatial dimensions, and is used in the decoder part of the autoencoder.
- **Final Conv2D Layer**: Outputs the denoised image with one channel (grayscale), using a sigmoid activation function to produce values between 0 and 1.

### 7. **Compiling and Training the Model**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_noisy, X_train, epochs=5, batch_size=256, validation_data=(x_test_noisy, X_test))
```

- **Compilation**: The model is compiled with the Adam optimizer and binary cross-entropy loss function. Since this is a denoising task, binary cross-entropy is used, which is typical for tasks like image reconstruction where pixel values are between 0 and 1.
- **Training**: The model is trained for 5 epochs using noisy images (`x_train_noisy`) as input and original images (`X_train`) as the target (ground truth). The batch size is set to 256, and validation data is provided using `x_test_noisy` and `X_test`.

### 8. **Saving the Model**

```python
model.save('./denoise_2.keras')
```

- **Saving the Model**: After training, the model is saved to the specified path (`denoise_2.keras`) for later use or evaluation.

### 9. **Denoising the Test Images**

```python
denoised_images = model.predict(x_test_noisy)
```

- **Prediction**: The trained autoencoder is used to predict (or denoise) the noisy test images (`x_test_noisy`).

### 10. **Plotting the Results**

```python
n = 5  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 56), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 56), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].reshape(28, 56), cmap='gray')
    plt.title("Denoised")
    plt.axis("off")

plt.show()
```

- **Visualization**: A set of 5 test images is displayed in three rows:
  - The first row shows the original images.
  - The second row shows the noisy versions of the images.
  - The third row shows the denoised versions produced by the trained autoencoder.

### 11. **Model Training Output**

```text
Epoch 1/5
235/235 ━━━━━━━━━━━━━━━━━━━━ 24s 93ms/step - accuracy: 0.7960 - loss: 0.3472 - val_accuracy: 0.8048 - val_loss: 0.1483
Epoch 2/5
235/235 ━━━━━━━━━━━━━━━━━━━━ 22s 93ms/step - accuracy: 0.8061 - loss: 0.1457 - val_accuracy: 0.8075 - val_loss: 0.1348
...
```

- **Training Output**: The model's

 performance (accuracy and loss) is displayed for each epoch. This helps track how well the model is learning and improving over time.

### Conclusion

The code demonstrates how to use a convolutional autoencoder for denoising images. It trains on the MNIST dataset by adding noise to the images and then uses the trained model to denoise them, visualizing the results in a structured manner.


# In[ ]:


Here are the detailed interpretations of each step and result in the code:

### 1. **Importing Libraries**

- **NumPy** and **Matplotlib**: These libraries are standard for data manipulation and visualization in Python. They are crucial for handling image data (NumPy) and displaying it (Matplotlib).
- **TensorFlow/Keras**: TensorFlow is the backend for deep learning models, and Keras (as part of TensorFlow) provides a higher-level API for building and training models. It simplifies the process of constructing a neural network, especially with layers like `Conv2D`, `MaxPool2D`, and `UpSampling2D`.

### 2. **Loading the Dataset**

- The **MNIST dataset** is a widely used dataset of handwritten digits (28x28 grayscale images). It's commonly used for training and testing image classification models. The code loads the dataset (`train_images`, `test_images`, `train_labels`, `test_labels`) into NumPy arrays.
  - **Training Images (`X_train`)**: These are the images the model will learn from.
  - **Test Images (`X_test`)**: These are the images used to evaluate the model's performance after training.
  - **Labels (`y_train`, `y_test`)**: The corresponding labels (the actual digit for each image) are not used directly in this task since this is a denoising problem.

### 3. **Data Normalization and Noise Addition**

- **Normalization**: The pixel values are normalized to the range `[0, 1]` by dividing by 255. This is a standard preprocessing step in deep learning, as it helps models converge faster and avoids issues with large input values.
  - **Before Normalization**: The pixel values are in the range `[0, 255]`.
  - **After Normalization**: The pixel values are scaled to the range `[0, 1]`, which is more suitable for neural network input.

- **Adding Gaussian Noise**:
  - **Noise Factor**: The `noise_factor = 0.6` controls the strength of the noise. A larger factor would result in more noise being added to the images.
  - **Gaussian Noise**: Random noise with a mean of 0 and a standard deviation determined by the `noise_factor` is added to the images. This simulates a noisy environment (e.g., sensor noise, transmission noise) that the autoencoder needs to learn to remove.
  - **Clipping**: The `np.clip()` function ensures the noisy pixel values remain within the valid range `[0, 1]`. Without clipping, the noise might push some pixel values outside of this range, which could cause errors during model training.

### 4. **Reshaping the Data for the Model**

- **Reshaping**: The images are reshaped into 4D arrays with dimensions `(num_images, height, width, channels)` to fit the input shape required by the Keras model.
  - **`X_train.shape = (num_images, 28, 56, 1)`**: This shape indicates 28x56 pixel images (possibly due to resizing, though the MNIST images are typically 28x28) with one channel (grayscale).
  - The reshaped images are necessary because Keras models expect the input to be in this format for convolutional operations.

### 5. **Displaying Original and Noisy Images**

- The first image from the training set is displayed in two forms: the original clean image and the noisy version.
  - **Original Image**: This is the clean MNIST digit image without any noise added.
  - **Noisy Image**: This is the same image after Gaussian noise has been added.
  
  **Interpretation**: By displaying these images side by side, we can visually assess the impact of noise on the quality of the images. The noisy image looks visibly degraded compared to the original image.

### 6. **Building the Autoencoder Model**

- **Convolutional Autoencoder**: The architecture consists of several convolutional layers and pooling layers to first compress the image into a smaller latent space and then reconstruct it back to the original size.
  - **Encoder**:
    - The first two `Conv2D` layers extract features from the noisy images, reducing spatial dimensions using `MaxPool2D`.
    - The convolutional filters capture low-level features like edges and textures from the image.
  - **Decoder**:
    - The `UpSampling2D` layers gradually increase the image dimensions.
    - The final `Conv2D` layer reconstructs the image into a denoised version, with a single output channel (grayscale) using a sigmoid activation function to scale the output to the range `[0, 1]`.

**Interpretation**: This structure ensures that the model can learn both the important features (through convolution) and how to reconstruct the image (through upsampling). The architecture is designed for image denoising, with the encoder capturing noisy features and the decoder learning to remove them.

### 7. **Compiling and Training the Model**

- **Compiling the Model**:
  - **Optimizer**: Adam is a commonly used optimization algorithm that combines the advantages of both SGD (Stochastic Gradient Descent) and RMSProp. It adjusts the learning rate during training to improve convergence.
  - **Loss Function**: Binary Cross-Entropy loss is used because the task involves predicting pixel values between 0 and 1. It measures how well the predicted denoised image matches the original clean image.
  - **Metrics**: Accuracy is tracked during training to monitor how well the model is performing.

- **Training the Model**:
  - The model is trained for 5 epochs, which is relatively short, but often enough for tasks like denoising.
  - The training data (`x_train_noisy`) is used as input, and the original clean images (`X_train`) are the target output.
  - **Validation**: The model is validated on the test set during training to check for overfitting. The `validation_data` argument allows the model to be evaluated after each epoch on unseen noisy images (`x_test_noisy`) and their clean counterparts (`X_test`).

### 8. **Saving the Model**

- After training, the model is saved to disk using `model.save()`. This allows the model to be loaded later for inference or further fine-tuning without retraining from scratch.

**Interpretation**: Saving the model is an important step as it allows us to preserve the trained model and use it for real-world applications, such as denoising noisy images.

### 9. **Denoising the Test Images**

- The trained autoencoder model is used to predict (denoise) the noisy test images (`x_test_noisy`).
  
**Interpretation**: This step evaluates how well the trained model generalizes to new, unseen data. The output (`denoised_images`) should be a cleaner version of the noisy test images.

### 10. **Plotting the Results**

- **Plotting the Results**: The code visualizes a subset of 5 test images:
  - **Original**: The clean, original images from the test set.
  - **Noisy**: The noisy versions of the test images, showing the added Gaussian noise.
  - **Denoised**: The images produced by the trained autoencoder, which should have the noise removed.

**Interpretation**:
- **Original vs Noisy**: The difference between the clean and noisy images is visually obvious. The noise disturbs the clarity of the images, which makes them harder to interpret.
- **Noisy vs Denoised**: The denoised images should look much closer to the original images, with the noise significantly reduced. This demonstrates the model's effectiveness in removing noise, showing that it has learned the underlying structure of the clean images.

### 11. **Model Training Output**

The training logs show the performance of the model during each epoch:
- **Accuracy**: The model starts with an accuracy of 79.6% on the training set in the first epoch and improves slightly to 80.94% by the fifth epoch. This indicates that the model is effectively learning to denoise the images.
- **Loss**: The loss starts high (0.3472) but decreases over time (to 0.1270). This suggests that the model is gradually minimizing the difference between the predicted denoised images and the original images.

**Interpretation**:
- The relatively steady improvement in accuracy and loss indicates that the autoencoder is successfully learning to denoise the images.
- The relatively high final accuracy (around 80%) suggests that the model is fairly effective in reconstructing the clean version of the noisy images.

### Conclusion

The code demonstrates how to train a convolutional autoencoder to denoise images. By adding Gaussian noise to MNIST images and using a convolutional autoencoder to remove this noise, the model learns to extract important features from noisy data and reconstruct cleaner versions of the images. The visualizations and training logs show that the model is successful in its task, as evidenced by the denoised images and the improvement in loss and accuracy during training.


