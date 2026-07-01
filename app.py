"""
Thai Fruit Recognizer - Streamlit Web App
------------------------------------------
รัน:  streamlit run app.py

ต้องมีไฟล์ (อยู่โฟลเดอร์เดียวกับ app.py):
  - thai_fruit_model.keras
  - class_names.json
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# ---------- Config ----------
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "thai_fruit_model.keras"
LABELS_PATH = BASE / "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Thai Fruit Recognizer", page_icon="🍈", layout="centered")


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def predict(model, class_names, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = probs.argsort()[::-1][:3]
    return [(class_names[i], float(probs[i]) * 100) for i in idx]


# ---------- UI ----------
st.title("🍈 Thai Fruit Recognizer")
st.caption("อัปโหลดรูปผลไม้ แล้วให้ AI ทายว่าเป็นผลไม้ชนิดใด (MobileNetV2 · 6 คลาส)")

try:
    model, class_names = load_assets()
except Exception as e:
    st.error(
        "โหลดโมเดลไม่ได้ — ตรวจว่ามี `thai_fruit_model.keras` และ "
        f"`class_names.json` อยู่โฟลเดอร์เดียวกับ app.py\n\n{e}"
    )
    st.stop()

with st.sidebar:
    st.subheader("ผลไม้ที่รู้จัก")
    for c in class_names:
        st.write(f"• {c}")

file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png", "bmp", "webp"])

if file:
    img = Image.open(file)
    st.image(img, caption="รูปที่เลือก", use_container_width=True)

    if st.button("🔍 วิเคราะห์", type="primary"):
        with st.spinner("กำลังวิเคราะห์..."):
            results = predict(model, class_names, img)

        top_label, top_conf = results[0]
        st.success(f"### {top_label}  ({top_conf:.1f}%)")

        st.write("**อันดับความเป็นไปได้:**")
        for label, conf in results:
            st.write(label)
            st.progress(min(int(conf), 100))
            st.caption(f"{conf:.2f}%")
else:
    st.info("👆 อัปโหลดรูปผลไม้เพื่อเริ่มวิเคราะห์")