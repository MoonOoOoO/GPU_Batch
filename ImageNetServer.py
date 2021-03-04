# import time 
import tensorflow as tf
import cv2
import numpy as np
import tensorflow.python.compiler.tensorrt.trt_convert as trt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from flask import Flask, request
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
app = Flask(__name__)
img_path = 'elephant.jpg'

model = ResNet50(weights='imagenet')

# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# for i in range(100):
#     # tic = time.time()
#     pre = model.predict(x)
#     # print("--- %s ms ---" % ((time.time() - tic) * 1000))
#     # print(decode_predictions(pre)[0])
for i in range(batch_size):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batched_input[i, :] = x
batched_input = tf.constant(batched_input)

preds = model.predict(batched_input)


def prepare_image(raw_image):
    img = cv2.resize(raw_image, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


@app.route('/')
def index():
    return "hello world"


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        image_file = request.files['file'].read()
        if image_file:
            image_array = np.fromstring(image_file, np.uint8)
            image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            x = prepare_image(image_decode)
            predicts = model.predict(x)
            results = decode_predictions(predicts)
            return str(results)


if __name__ == '__main__':
    print("start")
    #    app.run('0.0.0.0')
