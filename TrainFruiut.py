from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = VGG16(include_top=False, input_shape=(224, 224, 3))

for layer in model.layers:
	layer.trainable = False

flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
model = Model(inputs=model.inputs, outputs=output)

opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(featurewise_center=True)
datagen.mean = [123.68, 116.779, 103.939]
train_it = datagen.flow_from_directory('D:\scr/AI_train_fruit/archive', class_mode='binary',
	batch_size=16, target_size=(224, 224))
test_it = datagen.flow_from_directory('D:\scr/AI_train_fruit/archive', class_mode='binary',
	batch_size=16, target_size=(224, 224))
model.fit(train_it, steps_per_epoch=len(train_it),validation_data=test_it,	epochs=90, verbose=1)

model.save('D:/scr/AI_train_fruit/fruit.h5')