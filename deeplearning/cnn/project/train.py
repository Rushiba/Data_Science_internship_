# train.py
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical

print("🚀 Loading CIFAR-10 Dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Trim dataset slightly for ultra-fast training in VS Code
x_train, y_train = x_train[:15000], y_train[:15000]

# Normalize values to [-1, 1] for MobileNetV2
x_train = (x_train.astype('float32') / 127.5) - 1.0
x_test = (x_test.astype('float32') / 127.5) - 1.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("🧠 Building High-Accuracy Transfer Model (96x96 internal scaling)...")
# Using a 96x96 feature map internally prevents any downsampling dimension errors
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = Sequential([
    tf.keras.layers.UpSampling2D(size=(3,3), input_shape=(32, 32, 3)), # Upscale 32x32 to 96x96
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🏋️ Training model...")
model.fit(x_train, y_train_cat, epochs=3, batch_size=64, validation_split=0.1)

# Get current folder path to prevent any path issues
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_path = os.path.join(current_dir, 'my_model.h5')
meta_path = os.path.join(current_dir, 'model_meta.joblib')

print(f"\n💾 Saving assets to: {current_dir}")
model.save(model_path)

metadata = {
    'class_names': class_names,
    'input_shape': (32, 32, 3)
}
joblib.dump(metadata, meta_path)
print("🎉 Model trained and files generated successfully!")
