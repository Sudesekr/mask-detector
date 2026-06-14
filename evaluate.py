import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model.h5"
IMG_SIZE = 224
BATCH = 32
DATA_DIR = "dataset/test"

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    shuffle=False
)

preds = model.predict(test_gen)
y_pred = (preds.flatten() >= 0.5).astype(int)
y_true = test_gen.classes

print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
