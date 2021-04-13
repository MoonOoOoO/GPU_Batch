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
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 50
BATCH_TIMEOUT = 1
CHECK_INTERVAL = 0.01

model = ResNet50(weights='imagenet')

requests_queue = Queue()

app = Flask(__name__)


def prepare_image(raw_image):
    img = cv2.resize(raw_image, dsize=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def form_batch(requests_batch):
    batched_input = np.zeros((len(requests_batch), 224, 224, 3), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        results = [executor.submit(prepare_image, img['input']) for img in requests_batch]
    for i, x in zip(range(BATCH_SIZE), results):
        batched_input[i, :] = x.result()
    return batched_input


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) >= BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

        batched_input = form_batch(requests_batch)
        batch_outputs = model.predict(batched_input, batch_size=BATCH_SIZE)

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
