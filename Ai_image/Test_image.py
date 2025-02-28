import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# โหลดโมเดล AI
model = tf.keras.models.load_model('thai_fruit_model.h5')
class_names = ['jackfruit', 'long kong','Muntingia', 'Pineapple', 'rambutan', 'sapodilla']

# ฟังก์ชันเลือกรูปภาพ
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        load_image(file_path)

# ฟังก์ชันโหลดและแสดงภาพ
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    global image_path
    image_path = file_path

# ฟังก์ชันทำนายผลไม้
def recognize_image():
    if not image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return

    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    fruit_name = class_names[predicted_class]
    result_label.configure(text=f"🍍 ผลไม้ที่ทำนาย: {fruit_name}\n🎯 ความมั่นใจ: {confidence:.2f}%")

# สร้างหน้าต่าง GUI
ctk.set_appearance_mode("light")  # โหมดแสง
ctk.set_default_color_theme("blue")  # ธีมสี

window = ctk.CTk()
window.title("🌿 AI Fruit Recognizer")
window.geometry("450x600")

# พื้นหลัง Gradient
bg_frame = ctk.CTkFrame(window, fg_color=("Yellow", "Yellow"))
bg_frame.pack(fill="both", expand=True)

# หัวข้อโปรแกรม
title_label = ctk.CTkLabel(bg_frame, text="🍉 โปรแกรมแยกประเภทผลไม้ 🍌", text_color="black", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# ส่วนแสดงภาพ
panel = ctk.CTkLabel(bg_frame, text="", width=250, height=250, corner_radius=10, fg_color="white")
panel.pack(pady=20)

# ปุ่มเลือกไฟล์
browse_button = ctk.CTkButton(bg_frame, text="📂 เลือกรูปภาพ", command=browse_image, fg_color="#FFA500", text_color="white", font=("Arial", 14, "bold"))
browse_button.pack(pady=10)

# ปุ่มทำนายผลไม้
recognize_button = ctk.CTkButton(bg_frame, text="🔍 ทำนายผลไม้", command=recognize_image, fg_color="#008CBA", text_color="white", font=("Arial", 14, "bold"))
recognize_button.pack(pady=10)

# แสดงผลลัพธ์
result_label = ctk.CTkLabel(bg_frame, text="🔽 รอทำนายผล 🔽", text_color="black", font=("Arial", 16, "bold"))
result_label.pack(pady=20)

# ตัวแปรเก็บพาธรูปภาพ
image_path = None

window.mainloop()
