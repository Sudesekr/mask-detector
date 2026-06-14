import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Daha güvenilir yüz tespiti için MTCNN yüklü ise kullan
try:
    from mtcnn import MTCNN
    detector = MTCNN()
    use_mtcnn = True
except Exception:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    use_mtcnn = False

IMG_SIZE = 224
LABELS = {0: "WITHOUT MASK", 1: "WITH MASK"}
COLORS = {0:(0,0,255), 1:(0,255,0)}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = []
    if use_mtcnn:
        results = detector.detect_faces(img_rgb)
        for r in results:
            x, y, w, h = r['box']
            x, y = max(0,x), max(0,y)
            faces.append((x,y,w,h))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_c = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        faces = faces_c

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0: continue
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

    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
