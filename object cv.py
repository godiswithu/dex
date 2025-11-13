#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the desired model (e.g., yolov8s.pt, yolov8m.pt)

# Load an image for detection
image_path = "206_png.rf.b0482e388ee10e72bbe47fe6d1e601f7.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Display results
annotated_image = results[0].plot()  # Annotate detections on the image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the annotated image
cv2.imwrite("output.jpg", annotated_image)


# In[ ]:




