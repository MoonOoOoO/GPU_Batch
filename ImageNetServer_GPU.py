import cv2
import time
import threading
import numpy as np
import tensorflow as tf

from queue import Empty, Queue
from flask import Flask, request as flask_request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

BATCH_SIZE = 31
BATCH_TIMEOUT = 2
CHECK_INTERVAL = 0.01

model = ResNet50(weights='imagenet')

requests_queue = Queue()

app = Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def prepare_image(raw_image):
    img = cv2.resize(raw_image, dsize=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) > BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

        batched_input = np.zeros((len(requests_batch), 224, 224, 3), dtype=np.float32)
        for i in range(len(requests_batch)):
            batched_input[i, :] = requests_batch[i]['input']

        batch_outputs = model.predict(tf.constant(batched_input), batch_size=BATCH_SIZE)
        decode_results = decode_predictions(batch_outputs)

        for request, output in zip(requests_batch, decode_results):
            request['output'] = str(output)


threading.Thread(target=handle_requests_by_batch).start()


@app.route('/predict', methods=['POST'])
def predict():
    if flask_request.method == 'POST':
        f = flask_request.files['file']
        if f:
            img = np.fromstring(f.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = prepare_image(img)
            request = {'input': img, 'time': time.time()}
            requests_queue.put(request)

            while 'output' not in request:
                time.sleep(CHECK_INTERVAL)
            return {'predictions': request['output']}


@app.route('/test', methods=['POST'])
def test():
    f = flask_request.files['file']
    img = np.fromstring(f.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = prepare_image(img)
    pred = model.predict(img)
    return str(decode_predictions(pred))


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
