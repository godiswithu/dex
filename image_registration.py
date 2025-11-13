#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessing:
    def __init__(self, image_path1, image_path2):
        self.image1 = cv2.imread(image_path1)
        self.image2 = cv2.imread(image_path2)
        if self.image1 is None or self.image2 is None:
            print("Error: Unable to load images.")
            exit()

    def display_images(self, images, titles):
        """Display a list of images with corresponding titles."""
        for img, title in zip(images, titles):
            plt.figure()
            plt.imshow(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()

    def rotate_image(self):
        """Rotate the image by a specified angle."""
        angle = float(input("Enter rotation angle in degrees: "))
        h, w = self.image1.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image1, M, (w, h))
        self.display_images([self.image1, rotated], ['Original Image', f'Rotated {angle}°'])

    def reflect_image(self):
        """Reflect the image vertically."""
        reflected = cv2.flip(self.image1, 0)
        self.display_images([self.image1, reflected], ['Original Image', 'Reflected Image'])

    def scale_image(self):
        """Scale the image to 70% of its original size."""
        scaled = cv2.resize(self.image1, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
        self.display_images([self.image1, scaled], ['Original Image', 'Scaled Image'])

    def crop_image(self):
        """Crop the image to remove a border of 100px from top-left and 30px from bottom-right."""
        x_start, y_start = 100, 100
        x_end, y_end = self.image1.shape[1] - 30, self.image1.shape[0] - 30
        cropped = self.image1[y_start:y_end, x_start:x_end]
        self.display_images([self.image1, cropped], ['Original Image', 'Cropped Image'])

    def affine_transform(self):
        """Apply affine transformation to align two images."""
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[60, 60], [210, 50], [60, 210]])
        M = cv2.getAffineTransform(pts1, pts2)
        aligned = cv2.warpAffine(self.image1, M, (self.image2.shape[1], self.image2.shape[0]))
        self.display_images([self.image2, aligned], ['Reference Image', 'Aligned Image'])

def main_menu():
    """Display the main menu options."""
    print("\n=== Image Processing Menu ===")
    print("1. Rotate Image")
    print("2. Reflect Image")
    print("3. Scale Image")
    print("4. Crop Image")
    print("5. Affine Transformation")
    print("6. Exit")

if __name__ == "__main__":
    processor = ImageProcessing('pothole.jpg', 'Pothole2.png')
    while True:
        main_menu()
        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            processor.rotate_image()
        elif choice == '2':
            processor.reflect_image()
        elif choice == '3':
            processor.scale_image()
        elif choice == '4':
            processor.crop_image()
        elif choice == '5':
            processor.affine_transform()
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


# In[ ]:


Let's break this code into meaningful blocks and explain each section in detail. This program allows users to perform several transformations and operations on images using OpenCV. We'll also cover the related concepts as we go.

---

### **1. Importing Libraries**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

#### **Explanation of Libraries**:
1. **cv2 (OpenCV)**:
   - OpenCV is a widely used library for image and video processing.
   - In this code, it is used to load images, perform transformations (rotation, scaling, etc.), and display processed images.

2. **numpy (np)**:
   - NumPy is used for handling numerical computations.
   - Here, it is used for creating and manipulating points for affine transformations.

3. **matplotlib.pyplot (plt)**:
   - Matplotlib is used for visualizing the images.
   - Unlike OpenCV's `imshow`, `plt.imshow` can handle color channels properly and is more suited for Python scripts.

---

### **2. `ImageProcessing` Class**
This class encapsulates all the image processing functionalities. It has methods for loading images, displaying them, and applying various transformations.

#### **Constructor (`__init__`)**
```python
class ImageProcessing:
    def __init__(self, image_path1, image_path2):
        self.image1 = cv2.imread(image_path1)
        self.image2 = cv2.imread(image_path2)
        if self.image1 is None or self.image2 is None:
            print("Error: Unable to load images.")
            exit()
```

- **Purpose**:
  - To load two images from the provided file paths (`image_path1` and `image_path2`).
  - If either image fails to load (e.g., incorrect file path), the program exits with an error message.

- **Concepts**:
  - **`cv2.imread()`**:
    - Reads an image from the specified file path.
    - If it fails, it returns `None`, so we check for this condition.

---

#### **Displaying Images (`display_images` Method)**
```python
def display_images(self, images, titles):
    for img, title in zip(images, titles):
        plt.figure()
        plt.imshow(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
```

- **Purpose**:
  - To display one or more images with titles using Matplotlib.
  
- **Key Steps**:
  1. **Color conversion**:
     - OpenCV uses the BGR color format, while Matplotlib uses RGB.
     - The line `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` ensures correct color representation.
  2. **Gray images**:
     - If an image is grayscale (`len(img.shape) == 2`), no color conversion is applied.
  3. **Visualization**:
     - `plt.title(title)` displays the title for each image.
     - `plt.axis('off')` hides axis markings for better visualization.

---

### **3. Image Transformations**

#### **(a) Rotating the Image**
```python
def rotate_image(self):
    angle = float(input("Enter rotation angle in degrees: "))
    h, w = self.image1.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(self.image1, M, (w, h))
    self.display_images([self.image1, rotated], ['Original Image', f'Rotated {angle}°'])
```

- **Concepts**:
  1. **Rotation matrix**:
     - `cv2.getRotationMatrix2D(center, angle, scale)`:
       - `center`: The pivot point for rotation (here, the image center).
       - `angle`: Rotation angle in degrees (positive for counterclockwise, negative for clockwise).
       - `scale`: Scaling factor (1.0 means no scaling).
  2. **Affine transformation**:
     - Rotation is a type of affine transformation applied using `cv2.warpAffine`.

- **Key Idea**:
  - The image is rotated around its center by the user-specified angle.

---

#### **(b) Reflecting the Image**
```python
def reflect_image(self):
    reflected = cv2.flip(self.image1, 0)
    self.display_images([self.image1, reflected], ['Original Image', 'Reflected Image'])
```

- **Concepts**:
  - **Reflection**:
    - `cv2.flip(image, flipCode)`:
      - `flipCode = 0`: Reflects vertically (upside-down).
      - `flipCode = 1`: Reflects horizontally (mirrored).
      - `flipCode = -1`: Reflects both vertically and horizontally.

---

#### **(c) Scaling the Image**
```python
def scale_image(self):
    scaled = cv2.resize(self.image1, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    self.display_images([self.image1, scaled], ['Original Image', 'Scaled Image'])
```

- **Concepts**:
  - **Resizing**:
    - `cv2.resize(image, dsize, fx, fy, interpolation)`:
      - `dsize`: Output size. Here, it is `None` since scaling factors (`fx` and `fy`) are provided.
      - `fx`, `fy`: Scaling factors along the x and y axes, respectively.
      - `interpolation`: Method for resizing:
        - `cv2.INTER_LINEAR`: Bilinear interpolation (good for downscaling).

---

#### **(d) Cropping the Image**
```python
def crop_image(self):
    x_start, y_start = 100, 100
    x_end, y_end = self.image1.shape[1] - 30, self.image1.shape[0] - 30
    cropped = self.image1[y_start:y_end, x_start:x_end]
    self.display_images([self.image1, cropped], ['Original Image', 'Cropped Image'])
```

- **Concepts**:
  - **Cropping**:
    - Achieved by slicing the NumPy array representing the image.
    - `[y_start:y_end, x_start:x_end]`: Specifies the region of interest.

---

#### **(e) Affine Transformation**
```python
def affine_transform(self):
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[60, 60], [210, 50], [60, 210]])
    M = cv2.getAffineTransform(pts1, pts2)
    aligned = cv2.warpAffine(self.image1, M, (self.image2.shape[1], self.image2.shape[0]))
    self.display_images([self.image2, aligned], ['Reference Image', 'Aligned Image'])
```

- **Concepts**:
  - **Affine Transformation**:
    - Maps three points (`pts1`) on the original image to corresponding points (`pts2`) on the output image.
    - **`cv2.getAffineTransform(pts1, pts2)`**:
      - Computes the affine transformation matrix.
    - **`cv2.warpAffine(image, M, size)`**:
      - Applies the affine transformation.

---

### **4. Main Menu**
```python
def main_menu():
    print("\n=== Image Processing Menu ===")
    print("1. Rotate Image")
    print("2. Reflect Image")
    print("3. Scale Image")
    print("4. Crop Image")
    print("5. Affine Transformation")
    print("6. Exit")
```

- **Purpose**:
  - Provides a menu interface for selecting operations.

---

### **5. Main Execution**
```python
if __name__ == "__main__":
    processor = ImageProcessing('pothole.jpg', 'Pothole2.png')
    while True:
        main_menu()
        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            processor.rotate_image()
        elif choice == '2':
            processor.reflect_image()
        elif choice == '3':
            processor.scale_image()
        elif choice == '4':
            processor.crop_image()
        elif choice == '5':
            processor.affine_transform()
        elif choice == '6':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
```

- **Purpose**:
  - Implements a menu-driven program for user interaction.
  - Calls the appropriate method based on user input.

- **Dynamic Behavior**:
  - Users provide inputs (e.g., rotation angle) dynamically during runtime.

---

### **Key Concepts in the Code**:
1. **Image Transformations**:
   - Rotation, scaling, reflection, cropping, and affine transformations are fundamental in image processing.

2. **Interactive Programming**:
   - The menu-driven interface makes the program user-friendly.

3. **Object-Oriented Programming (OOP)**:
   - Encapsulation of functionalities in a class improves code modularity and reusability.

4. **Visualization**:
   - Clear display of original and processed images helps understand the effects of each operation.

This program demonstrates a comprehensive approach to basic image processing tasks using OpenCV and Python.

