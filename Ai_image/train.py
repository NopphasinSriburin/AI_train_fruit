"""
Thai Fruit Classifier - Training (improved)
-------------------------------------------
โครงสร้างข้อมูลที่คาดหวัง:
    Ai_image/dataset/train/<class>/*.jpg
    Ai_image/dataset/test/<class>/*.jpg

จุดที่ปรับปรุงเพื่อความแม่น:
  1. Augmentation หนักขึ้น (flip, rotation, zoom, contrast, translation)
  2. Class weights - ชดเชยคลาสที่รูปน้อย (ปัญหารูปไม่เท่ากัน)
  3. Fine-tune 2 เฟส (freeze -> unfreeze 30 layer บนสุด)
  4. ReduceLROnPlateau - ลด learning rate อัตโนมัติเมื่อ val หยุดดีขึ้น
  5. เซฟทั้งโมเดล + class_names.json ให้ app.py ใช้ตรงกัน
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------- Config ----------
TRAIN_DIR = "Ai_image/dataset/train"
TEST_DIR  = "Ai_image/dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 15      # เฟส 1: เทรนเฉพาะหัว
EPOCHS_FINE = 10      # เฟส 2: fine-tune
MODEL_PATH = "thai_fruit_model.keras"
LABELS_PATH = "class_names.json"

# ---------- Data ----------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=True, seed=42,
)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False,
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# ---------- Class weights (ชดเชยคลาสรูปน้อย) ----------
counts = Counter()
for cls_idx, cls in enumerate(class_names):
    n = sum(1 for _ in Path(TRAIN_DIR, cls).glob("*")
            if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    counts[cls_idx] = n
    print(f"  train/{cls:15s}: {n} รูป")

total = sum(counts.values())
class_weight = {i: total / (num_classes * counts[i]) if counts[i] else 1.0
                for i in range(num_classes)}
print(f"Class weights: { {k: round(v,2) for k,v in class_weight.items()} }")

# ---------- Augmentation ----------
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE

def prep(ds, training=False):
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds = prep(train_ds_raw, training=True)
test_ds  = prep(test_ds_raw)

# ---------- Model ----------
base = MobileNetV2(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation="softmax")(x)
model = Model(base.input, output)

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

# ---------- เฟส 1: เทรนหัว ----------
print("\n===== PHASE 1: train head =====")
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_HEAD,
          class_weight=class_weight, callbacks=callbacks)

# ---------- เฟส 2: fine-tune 30 layer บนสุด ----------
print("\n===== PHASE 2: fine-tune =====")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_FINE,
          class_weight=class_weight, callbacks=callbacks)

# ---------- สรุป ----------
loss, acc = model.evaluate(test_ds)
print(f"\nFinal Test accuracy: {acc*100:.2f}%")
print(f"Saved model  -> {MODEL_PATH}")
print(f"Saved labels -> {LABELS_PATH}")