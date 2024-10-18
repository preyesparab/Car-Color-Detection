import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the saved model
model = load_model('my_model.keras')
model = tf.keras.models.load_model('my_model.keras')

# Compile the model with the required metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# cars.xml =r'D:\projects\car\cars.xml'
# haarcascade_fullbody.xml = r'haarcascade_fullbody.xml'
# Load Haar cascades
car_cascade = cv2.CascadeClassifier('cars.xml')
person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Label Binarizer classes (update these based on your trained classes)
lb_classes = ['black', 'blue', 'brown', 'gold','green', 'grey', 'orange', 'pink','purple', 'red', 'silver', 'tan','white', 'yellow', 'beige']  
global img, loaded_img

# Define maximum window and image size
MAX_WINDOW_WIDTH = 800
MAX_WINDOW_HEIGHT = 600

def resize_image(image, max_width, max_height):
    """Resizes the image to fit within the given max dimensions while maintaining aspect ratio."""
    height, width, _ = image.shape
    
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if width > height:
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image
    
    return resized_image

def detect_cars_and_people(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    
    # Detect people
    persons = person_cascade.detectMultiScale(gray, 1.1, 3)
    
    # Process detected cars
    for (x, y, w, h) in cars:
        car_image = image[y:y+h, x:x+w]
        car_image_resized = cv2.resize(car_image, (224, 224))
        car_image_array = np.expand_dims(car_image_resized, axis=0)
        
        # Predict the car color
        prediction = model.predict(car_image_array)
        color = lb_classes[np.argmax(prediction)]
        
        # Draw bounding boxes based on the car color
        if color == 'blue':
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for blue cars
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for other cars

    # Process detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for people
    
    return image, len(cars), len(persons)

def open_image():
    global loaded_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        loaded_img = img.copy()  # Save the original image for later processing
        resized_img = resize_image(img, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)
        
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        panel.config(image=img_tk)
        panel.image = img_tk

def run_detection():
    global loaded_img
    if loaded_img is not None:
        resized_img = resize_image(loaded_img, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)
        processed_img, car_count, person_count = detect_cars_and_people(resized_img)
        
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display updated image with bounding boxes
        panel.config(image=img_tk)
        panel.image = img_tk

        result_text.set(f"Cars detected: {car_count}, People detected: {person_count}")
    else:
        result_text.set("Please upload an image first.")

# Create the GUI window
root = tk.Tk()
root.title("Car Color Detection")

# Set a fixed window size
root.geometry(f"{MAX_WINDOW_WIDTH}x{MAX_WINDOW_HEIGHT}")

# Panel to display the image
panel = tk.Label(root)
panel.pack()

# Button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=open_image)
upload_button.pack()

# Button to run the detection
run_button = tk.Button(root, text="Run Detection", command=run_detection)
run_button.pack()

# Label to display the detection results
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack()

# Start the GUI
root.mainloop()
