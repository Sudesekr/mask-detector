import tensorflow as tf

# Keras modelimizi yükle
model = tf.keras.models.load_model("model.h5")

# TFLite converter oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizasyon ekle (modeli küçültür, mobilde hızlı çalışır)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Float16 veya integer quantization da eklenebilir (opsiyonel)
# converter.target_spec.supported_types = [tf.float16]

# TFLite modelini oluştur
tflite_model = converter.convert()

# Kaydet
tflite_path = "model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite modeli oluşturuldu: {tflite_path}")
