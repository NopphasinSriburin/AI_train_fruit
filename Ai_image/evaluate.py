"""
Thai Fruit Classifier - Evaluation
----------------------------------
วัดผลโมเดลที่เทรนแล้ว:
  - นับจำนวนรูปแต่ละคลาส (train + test)  [glob แก้ให้นับครบทุกนามสกุล]
  - เตือนถ้าคลาสไหน test ว่าง (support = 0)
  - Test accuracy
  - รายงานแยกคลาส (precision / recall / f1)  [ปิด warning ด้วย zero_division]
  - Confusion matrix (แถว=จริง, คอลัมน์=ทาย)

รัน:  python evaluate.py
ต้องมี:  pip install scikit-learn
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = "Ai_image/dataset/train"
TEST_DIR  = "Ai_image/dataset/test"
MODEL_PATH = "thai_fruit_model.keras"
IMG_SIZE = (224, 224)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(d):
    """นับรูปแบบครอบคลุมทุกนามสกุล (case-insensitive)"""
    out = {}
    for cls in sorted(Path(d).iterdir()):
        if cls.is_dir():
            out[cls.name] = sum(1 for f in cls.iterdir()
                                if f.suffix.lower() in IMG_EXT)
    return out

# ---------- นับจำนวนรูป ----------
print("=" * 55)
print("จำนวนรูปแต่ละคลาส")
print("=" * 55)
train_counts = count_images(TRAIN_DIR)
test_counts  = count_images(TEST_DIR)

all_classes = sorted(set(train_counts) | set(test_counts))
print(f"{'class':15s}{'train':>8s}{'test':>8s}")
for c in all_classes:
    tr = train_counts.get(c, 0)
    te = test_counts.get(c, 0)
    flag = "  <-- test ว่าง!" if te == 0 else ""
    print(f"{c:15s}{tr:>8d}{te:>8d}{flag}")

empty = [c for c in all_classes if test_counts.get(c, 0) == 0]
if empty:
    print(f"\n[!] คลาสที่ไม่มีรูป test: {empty}")
    print("    -> วัด recall คลาสนี้ไม่ได้ ควรย้าย/เพิ่มรูปเข้า test ก่อน")

# ---------- โหลด test set ----------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=16,
    label_mode="categorical", shuffle=False,
)
class_names = test_ds.class_names
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# ---------- ทำนาย ----------
model = tf.keras.models.load_model(MODEL_PATH)
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)

acc = (y_true == y_pred).mean() * 100
print("\n" + "=" * 55)
print(f"Test Accuracy: {acc:.2f}%")
print("=" * 55)

# ---------- รายงานแยกคลาส (ไม่มี warning) ----------
print("\nรายงานแยกคลาส (precision / recall / f1):")
print(classification_report(
    y_true, y_pred, target_names=class_names,
    labels=list(range(len(class_names))),
    digits=3, zero_division=0,
))

# ---------- Confusion matrix ----------
print("Confusion Matrix (แถว=จริง, คอลัมน์=ทาย):")
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
print(f"{'':15s}" + "".join(f"{c[:8]:>9s}" for c in class_names))
for i, row in enumerate(cm):
    print(f"{class_names[i]:15s}" + "".join(f"{v:>9d}" for v in row))

# ---------- ชี้คู่ที่สับสนที่สุด ----------
print("\nคู่ที่โมเดลสับสนมากสุด:")
pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            pairs.append((cm[i][j], class_names[i], class_names[j]))
pairs.sort(reverse=True)
if pairs:
    for n, a, b in pairs[:5]:
        print(f"  {a} ถูกทายเป็น {b}: {n} ครั้ง")
else:
    print("  ไม่มี - ทายถูกทุกใบ")