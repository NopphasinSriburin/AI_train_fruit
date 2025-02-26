import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Function to browse and select an image file
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        load_image(file_path)

# Function to load and display the selected image
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((200, 200))  # Resize image to fit the display
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img  # Store the image in the panel
    global image_path
    image_path = file_path

# Function to recognize the fruit from the selected image
def recognize_image():
    if not image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return

    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the index of the highest prediction
    confidence = np.max(predictions) * 100  # Get the confidence as a percentage

    # Display the result
    fruit_name = class_names[predicted_class]
    result_label.config(text=f"Predicted Fruit: {fruit_name}\nConfidence: {confidence:.2f}%")

# Load the trained AI model
model = tf.keras.models.load_model('thai_fruit_model.h5')

# Class names for fruits (English)
class_names = ['jackfruit', 'long kong', 'Pineapple', 'rambutan', 'sapodilla']

# Create the GUI window
window = tk.Tk()
window.title("AI Fruit Recognizer")

# Set the window size and background color
window.geometry("400x500")
window.config(bg="#f5f5f5")

# Panel to display the selected image
panel = tk.Label(window, bg="#f5f5f5")
panel.pack(padx=10, pady=10)

# Browse button to select an image
browse_button = tk.Button(window, text="Browse", command=browse_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
browse_button.pack(pady=10)

# Recognize button to predict the fruit
recognize_button = tk.Button(window, text="Recognize", command=recognize_image, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
recognize_button.pack(pady=10)

# Label to show the result of the prediction
result_label = tk.Label(window, text="Predicted Fruit: ", bg="#f5f5f5", font=("Arial", 14))
result_label.pack(pady=20)

# Variable to store the selected image path
image_path = None

# Start the GUI event loop
window.mainloop()
