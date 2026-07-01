# 🍈 Thai Fruit Recognizer

ระบบจำแนกผลไม้ไทยจากภาพด้วย Deep Learning (Transfer Learning บน MobileNetV2)
รองรับ 6 ชนิด: **jackfruit · long kong · Muntingia · pineapple · rambutan · sapodilla**

พัฒนาจาก baseline **90%** จนได้ **98.36%** ด้วย data augmentation, class weights และ fine-tuning
พร้อม Web App (Streamlit) ให้อัปโหลดรูปแล้วทายผลได้ทันที

---

## 📊 ผลลัพธ์ (Test Accuracy: 98.36%)

| | Baseline | Improved |
|---|---|---|
| Test accuracy | 90.16% | **98.36%** |
| Macro F1 | 0.90 | **0.983** |
| คลาสที่ F1 = 1.00 | 1 | **4** |
| จำนวนใบที่ทายพลาด | 6 / 61 | **1 / 61** |

รายงานแยกคลาส:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Muntingia | 1.000 | 1.000 | 1.000 | 11 |
| jackfruit | 1.000 | 1.000 | 1.000 | 10 |
| long kong | 1.000 | 0.900 | 0.947 | 10 |
| pineapple | 1.000 | 1.000 | 1.000 | 10 |
| rambutan | 1.000 | 1.000 | 1.000 | 10 |
| sapodilla | 0.909 | 1.000 | 0.952 | 10 |

> เหลือพลาดเพียง 1 ใบ: long kong ถูกทายเป็น sapodilla (ผิวน้ำตาลทรงกลมคล้ายกัน)
> หมายเหตุ: test set มี 61 รูป การพลาด 1 ใบ = ~1.6% ตัวเลขอาจขยับเมื่อ test set ใหญ่ขึ้น

---

## 🧠 ระบบทำงานยังไง (How it works)

ภาพรวมของ pipeline ตั้งแต่รูปภาพจนถึงคำทำนาย:

```
รูปภาพ (any size)
   │
   ▼
[1] Preprocess  →  resize 224×224  +  preprocess_input (ปรับค่าสีให้ตรงกับที่ MobileNetV2 ถูกเทรนมา)
   │
   ▼
[2] MobileNetV2 backbone (pre-trained ImageNet)
       ดึง "ลักษณะเด่น" ของภาพ เช่น ขอบ, สี, พื้นผิว, ลวดลาย
   │
   ▼
[3] Classification head (ส่วนที่เราเทรนเอง)
       GlobalAveragePooling → Dense(128) → Dropout → Dense(6, softmax)
   │
   ▼
[4] Output: ความน่าจะเป็นของแต่ละคลาส (รวมกัน = 100%)
       เช่น rambutan 97.3%, long kong 1.8%, ...
```

**หัวใจคือ Transfer Learning** — แทนที่จะสอนโมเดลให้ "มองภาพเป็น" ตั้งแต่ศูนย์ (ต้องใช้รูปเป็นแสนใบ)
เราหยิบ MobileNetV2 ที่เคยเรียนรู้จากภาพ 1.4 ล้านใบใน ImageNet มาแล้ว มันรู้จักขอบ/สี/พื้นผิวทั่วไปอยู่แล้ว
เราแค่สอน "ส่วนหัว" ให้เอาลักษณะเหล่านั้นมาแยกผลไม้ไทย 6 ชนิด จึงใช้รูปแค่ ~40 ใบต่อคลาสก็พอ

---

## 🔧 เทคนิคที่ใช้ และช่วยอะไร

| เทคนิค | ช่วยอะไร | ทำไมโปรเจกต์นี้ต้องใช้ |
|---|---|---|
| **Transfer Learning (MobileNetV2)** | ไม่ต้องเทรนจากศูนย์ ใช้รูปน้อยก็แม่น | dataset เรามีแค่ ~40 รูป/คลาส |
| **Data Augmentation** | สร้างภาพหลากหลาย (พลิก/หมุน/ซูม/ปรับแสง) จากรูปเดิม | กัน overfit เมื่อรูปน้อย ให้โมเดลทนต่อมุม/แสงที่ต่างกัน |
| **Class Weights** | ถ่วงน้ำหนักคลาสที่รูปน้อยให้โมเดลใส่ใจมากขึ้น | pineapple มี 29 รูป แต่ long kong มี 50 — ไม่สมดุล |
| **Fine-tuning 2 เฟส** | เฟสแรกเทรนหัว, เฟสสองปลดล็อก layer บนสุดของ backbone | ดันความแม่นจาก ~90% → 98% |
| **ReduceLROnPlateau** | ลด learning rate อัตโนมัติเมื่อโมเดลหยุดพัฒนา | ช่วยให้ลู่เข้าจุดที่ดีที่สุดแทนที่จะแกว่ง |
| **EarlyStopping** | หยุดเทรนเองเมื่อไม่ดีขึ้น + คืนค่าโมเดลที่ดีที่สุด | กันเทรนเกินจนจำข้อมูล (overfit) |

---

## 📁 ไฟล์ในโปรเจกต์ทำอะไร

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | **Web App (Streamlit)** — อัปโหลดรูป → แสดงผลทาย top-3 พร้อม confidence bar |
| `Ai_image/train.py` | **สคริปต์เทรนโมเดล** — โหลดข้อมูล, augment, เทรน 2 เฟส, เซฟ `.keras` + `class_names.json` |
| `Ai_image/evaluate.py` | **สคริปต์วัดผล** — นับรูปแต่ละคลาส, คำนวณ accuracy, confusion matrix, ชี้คู่ที่โมเดลสับสน |
| `thai_fruit_model.keras` | โมเดลที่เทรนเสร็จแล้ว (พร้อมใช้ ไม่ต้องเทรนซ้ำ) |
| `class_names.json` | รายชื่อคลาสเรียงตาม index ที่โมเดล output — ทำให้ app แปลงเลขเป็นชื่อผลไม้ได้ถูก |
| `requirements.txt` | รายการ library ที่ต้องติดตั้ง |

---

## 🗂️ โครงสร้างโปรเจกต์

```
AI_train_fruit/
├── app.py                     # Streamlit web app
├── requirements.txt
├── README.md
├── class_names.json           # รายชื่อคลาส (เรียงตาม index ของโมเดล)
├── thai_fruit_model.keras     # โมเดลที่เทรนแล้ว
├── .gitignore
├── Ai_image/
│   ├── train.py               # เทรนโมเดล
│   ├── evaluate.py            # วัดผล + confusion matrix
│   └── dataset/
│       ├── train/<class>/*.jpg
│       └── test/<class>/*.jpg
└── Download_image_Auto/
    └── Download_image.py      # สคริปต์ช่วยโหลดรูปมาทำ dataset
```

---

## 🚀 วิธีใช้งาน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. รัน Web App (ใช้โมเดลที่เทรนไว้แล้ว)

```bash
streamlit run app.py
```

เบราว์เซอร์จะเปิดที่ `http://localhost:8501` → อัปโหลดรูปผลไม้ → กดวิเคราะห์ → เห็นผล top-3

### 3. เทรนโมเดลใหม่เอง (ถ้าต้องการ)

```bash
python Ai_image/train.py       # สร้าง thai_fruit_model.keras + class_names.json
python Ai_image/evaluate.py    # วัดผล + confusion matrix
```

> **หมายเหตุเรื่อง path:** `app.py` คาดว่า `thai_fruit_model.keras` และ `class_names.json`
> อยู่โฟลเดอร์เดียวกับตัวมันเอง หากย้ายไฟล์ ให้แก้ `MODEL_PATH` / `LABELS_PATH` ใน `app.py`

---

## 🛠️ Tech Stack

Python · TensorFlow / Keras · MobileNetV2 · scikit-learn · Streamlit · Pillow

---

## 📈 ที่มาของโปรเจกต์ (Development story)

1. เริ่มจากโค้ดเดิมที่ใช้ VGG16 แบบ binary (2 คลาส) — ปรับเป็น MobileNetV2 multi-class (6 คลาส)
2. วัดผลครั้งแรกได้ 88% แต่พบ bug: สคริปต์นับรูป test คลาส Muntingia พลาด (นับเป็น 0)
3. แก้ bug แล้ววัดใหม่ = 90.16% (ค่าจริง)
4. เพิ่ม augmentation + class weights + fine-tuning → **98.36%**
5. ทำ Streamlit web app + จัดโครงสร้าง repo ให้ clone ไปรันได้ทันที