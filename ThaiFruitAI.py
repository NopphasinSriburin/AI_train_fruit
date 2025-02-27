import tkinter as tk
from tkinter import Canvas, filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

def load_model(self):
    model = load_model('thai_fruit_model.h5')
    return model

class ThaiFruitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Thai Fruit Recognizer")
        self.root.geometry("800x600")
        
        # Initialize the AI model
        self.model = self.load_model()

        # Create Browse button
        self.browse_button = Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        # Create Canvas to display the image
        self.image_canvas = Canvas(self.root, width=400, height=400, bg="gray")
        self.image_canvas.pack()

        # Create Recognize button
        self.recognize_button = Button(self.root, text="Recognize Fruit", command=self.recognize_fruit, state="disabled")
        self.recognize_button.pack(pady=10)

        # Label to display the result
        self.result_label = Label(self.root, text="", font=("Arial", 24))
        self.result_label.pack(pady=10)
        
        # Store the path of the selected image
        self.image_path = None

    def load_model(self):
        """
        Load the pre-trained ResNet50 model.
        For Thai fruit recognition, you should fine-tune this model with your dataset.
        """
        model = ResNet50(weights='imagenet')
        return model

    def browse_image(self):
        """
        Browse and select an image file.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.recognize_button.config(state="normal")
            self.result_label.config(text="")
    
    def display_image(self, path):
        """
        Display the selected image on the canvas.
        """
        # Open and resize the image
        img = Image.open(path)
        img = img.resize((400, 400))
        self.photo = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def preprocess_image(self, path):
        """
        Preprocess the image to feed into the model.
        """
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def recognize_fruit(self):
        """
        Use the AI model to recognize the fruit in the image.
        """
        if not self.image_path:
            return
        
        processed_image = self.preprocess_image(self.image_path)
        predictions = self.model.predict(processed_image)
        decoded_preds = decode_predictions(predictions, top=3)[0]
        
        # For demonstration, we'll display the top prediction
        if decoded_preds:
            # In a real scenario, map ImageNet classes to your Thai fruit classes
            label = decoded_preds[0][1].replace('_', ' ').title()
            confidence = decoded_preds[0][2] * 100
            result_text = f"Prediction: {label}\nConfidence: {confidence:.2f}%"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Unable to recognize the fruit.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThaiFruitRecognizer(root)
    root.mainloop()