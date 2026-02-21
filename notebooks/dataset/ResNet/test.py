from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import cv2


# Load your trained model
model = load_model("C:/Users/Anes/Downloads/test model/plant_disease_resnet50.h5")

# Build class labels
train_dir = "C:/Users/Anes/Downloads/archive (4)/plantvillage_split/train"
class_labels = sorted(os.listdir(train_dir))

print("Number of classes:", len(class_labels))
print("First 5 labels:", class_labels[:5])

# Path to test image
img_path = "C:/Users/Anes/Downloads/TP1_BI/tomato1.jfif"

# Step 1: Read image
img = cv2.imread(img_path)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green range (tuned)
lower_green = np.array([20, 20, 20])
upper_green = np.array([100, 255, 255])

# Mask green areas
mask = cv2.inRange(hsv, lower_green, upper_green)

# Clean mask
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply mask
result = cv2.bitwise_and(img, img, mask=mask)

# If mask is empty, fallback to original image
if np.sum(mask) == 0:
    print("⚠️ No leaf detected, using original image")
    result = img

# Resize to 224x224
result_resized = cv2.resize(result, (224, 224))
img_array = image.img_to_array(result_resized)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess for ResNet50
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print("Predicted class:", class_labels[predicted_class])

# Show top-3 predictions
top_indices = predictions[0].argsort()[-3:][::-1]
print("\nTop-3 predictions:")
for i in top_indices:
    print(f"{class_labels[i]}: {predictions[0][i]*100:.2f}%")

