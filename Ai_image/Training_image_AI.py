import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ตั้งค่าพาธของชุดข้อมูล
train_dir = 'dataset/train'  # ชุดข้อมูลฝึก
test_dir = 'dataset/test'    # ชุดข้อมูลทดสอบ

# การเตรียมข้อมูลด้วย ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # การปรับขนาดภาพให้ค่าอยู่ในช่วง [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# การสร้างโมเดล CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(6, activation='softmax') # 6 คลาส (5 ผลไม้ + 1 อื่นๆ)
])

# คอมไพล์โมเดล
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ฝึกโมเดล
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # จำนวนขั้นตอนในการฝึก (ขึ้นอยู่กับขนาดชุดข้อมูล)
    epochs=10,
    validation_data=test_generator,
    validation_steps=50  # จำนวนขั้นตอนในการทดสอบ
)

# บันทึกโมเดล
model.save('thai_fruit_model.h5')

# แสดงกราฟประสิทธิภาพการฝึก
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
