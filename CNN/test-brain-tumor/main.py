import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load saved model
model = load_model('brain_tumor_model.h5')

# Function to predict brain tumor
def predict_tumor(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Check if image is read successfully
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Resize image
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Convert predicted class to label
    labels = ['Yes', 'No']
    predicted_label = labels[predicted_class]

    return predicted_label

# Test the function
image_path = 'brain1.jpeg'  # Replace with your image path
prediction = predict_tumor(image_path)
if prediction is not None:
    print(f'Predicted label: {prediction}')
