import cv2
import time
import flask
import warnings
import threading
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from queue import Empty, Queue
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_eff

app = Flask(__name__)
BATCH_SIZE = 8
BATCH_TIMEOUT = 0.1
CHECK_INTERVAL = 0.01
requests_queue = Queue()
warnings.filterwarnings("ignore")
CLASSES = ["DWG", "GPS", "IRR", "MEAS", "NON", "SGN", "WAT", "BCP", "BOV", "CD0", "CD1", "CDM", "CDR",
           "CPCOL", "CPWAL", "LOCEX", "LOCIN", "ODM", "ODS", "OVCAN", "OVFRT"]


def f_beta(y_true, y_pred, threshold_shift=0):
    beta = 2
    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def specificity(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    tnr = tn / (tn + fp + K.epsilon())
    return tnr


def dummy_weighted_categorical_loss(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, -1)
    condition = K.greater(K.sum(y_true), 0)
    return K.switch(condition, loss, K.zeros_like(loss))


def dummy_weighted_loss(y_true, y_pred):
    return K.mean((K.binary_crossentropy(y_true, y_pred)))


def prepare_image(raw_image):
    image = cv2.resize(raw_image, dsize=(299, 299))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input_eff(image)
    return image


def decode_predictions(prediction_batch):
    if len(prediction_batch.shape) != 2 or prediction_batch.shape[1] != len(CLASSES):
        raise ValueError('`decode_predictions` expects a batch of predictions')

    results = []

    prediction = prediction_batch.copy()
    indices = np.argmax(prediction[:, :9], axis=1)
    prediction[:, :9] = 0
    for idx, i in enumerate(indices):
        prediction[idx, i] = 1
        #   if BCP: CPDMG, CPTYPE, and LOC
        if CLASSES[i] == 'BCP':
            # CPDMG
            j = np.argmax(prediction[idx, 9:13]) + 9
            prediction[idx, 9:13] = 0
            prediction[idx, j] = 1
            # CPTYPE
            j = np.argmax(prediction[idx, 13:15]) + 13
            prediction[idx, 13:15] = 0
            prediction[idx, j] = 1
            # LOC
            j = np.argmax(prediction[idx, 15:17]) + 15
            prediction[idx, 15:17] = 0
            prediction[idx, j] = 1
            # wipe BOV predictions
            prediction[idx][17:21] = 0
        # if BOV: OVDMG, OVANG, and LOC
        elif CLASSES[i] == 'BOV':
            # OVDMG
            j = np.argmax(prediction[idx, 17:19]) + 17
            prediction[idx, 17:19] = 0
            prediction[idx, j] = 1
            # OVANG
            j = np.argmax(prediction[idx, 19:21]) + 19
            prediction[idx, 19:21] = 0
            prediction[idx, j] = 1
            # LOC
            j = np.argmax(prediction[idx, 15:17]) + 15
            prediction[idx, 15:17] = 0
            prediction[idx, j] = 1
            # wipe BOV predictions
            prediction[idx, 9:15] = 0
        else:  # if not BCP or BOV, wipe all predictions
            prediction[idx, 9:] = 0

    for preds in prediction:
        for index, result in enumerate(preds):
            if result >= 0.5 and CLASSES[index] != "BOV" and CLASSES[index] != "BCP":
                results.append((CLASSES[index], result))
    return results


def handle_requests_by_batch():
    data = {"results": []}
    batched_input = np.zeros((BATCH_SIZE, 299, 299, 3), dtype=np.float32)
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

            # if len(requests_batch) < BATCH_SIZE:
            #     for req in requests_batch:
            #         raw_prediction = model.predict(req["input"])
            #         print(raw_prediction)
            #
            #         raw_prediction = np.concatenate(raw_prediction, axis=1)
            #         print(raw_prediction)
            #
            #         results = decode_predictions(raw_prediction)
            #         print(results)
            #         data['results'] = []
            #         for result in results:
            #             label, prob = result
            #             data['results'].append({
            #                 "label": label,
            #                 "probability": float(prob)
            #             })
            #         req['output'] = data
            # else:
            for i, req in zip(range(len(requests_batch)), requests_batch):
                batched_input[i, :] = req["input"]

            print(len(batched_input))

            batched_input = tf.constant(batched_input)
            preds = model.predict(batched_input)

            for raw_prediction, req in zip(preds, requests_batch):
                print(raw_prediction)
                raw_prediction = np.concatenate(np.array(raw_prediction), axis=1)
                results = decode_predictions(raw_prediction)
                data['results'] = []
                for result in results:
                    label, prob = result
                    data['results'].append({
                        "label": label,
                        "probability": float(prob)
                    })
                req['output'] = data


threading.Thread(target=handle_requests_by_batch).start()


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        image_file = request.files['file'].read()
        if image_file:
            image_array = np.fromstring(image_file, np.uint8)
            image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = prepare_image(image_decode)
            image_req = {"input": image, "time": time.time()}
            requests_queue.put(image_req)
            while "output" not in image_req:
                time.sleep(CHECK_INTERVAL)
            return flask.jsonify(image_req['output'])


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

model = load_model("./efficientnet_model.h5",
                   custom_objects={
                       'fbeta': f_beta,
                       'recall': recall,
                       'precision': precision,
                       'specificity': specificity,
                       "loss": dummy_weighted_categorical_loss,
                       "weighted_loss": dummy_weighted_loss
                   })
print("Model loaded, classification server listening at http://localhost:16000/predict")

if __name__ == '__main__':
    WSGIServer(('0.0.0.0', 16000), app).serve_forever()
