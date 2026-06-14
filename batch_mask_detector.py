import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
LABELS = {0: "WITHOUT MASK", 1: "WITH MASK"}
COLORS = {0:(0,0,255), 1:(0,255,0)}

# Test resimlerinin bulunduğu klasör
TEST_DIR = "dataset/test"
# İşlenen resimlerin kaydedileceği klasör
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Yüz tespiti için Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Tüm sınıflar ve resimler üzerinde dön
for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_arr = face_resized.astype("float32") / 255.0
            face_arr = np.expand_dims(face_arr, axis=0)
            pred = model.predict(face_arr)[0][0]
            label = 1 if pred >= 0.5 else 0
            confidence = pred if label==1 else 1-pred
            color = COLORS[label]
            text = f"{LABELS[label]} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Sonucu kaydet
        cv2.imwrite(os.path.join(output_class_dir, img_name), frame)

print(f"Tüm test resimleri işlendi ve '{OUTPUT_DIR}' klasörüne kaydedildi.")
