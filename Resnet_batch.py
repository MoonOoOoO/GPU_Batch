import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

BATCH_SIZE = 32
img_path = 'elephant.jpg'

model = ResNet50(weights='imagenet')

for n in range(10):
    batched_input = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
    for i in range(BATCH_SIZE):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[i, :] = x
    batched_input = tf.constant(batched_input)
    preds = model.predict(batched_input, verbose=1, batch_size=BATCH_SIZE)
    batch_results = decode_predictions(preds)
    print(batch_results)


def prepare_image(raw_image):
    img = cv2.resize(raw_image, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
