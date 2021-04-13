import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 32
img_path = 'elephant.jpg'

model = ResNet50(weights='imagenet')


def preprocess_image(count):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def form_batch():
    batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = [executor.submit(preprocess_image, param) for param in range(BATCH_SIZE)]
        for i, x in zip(range(BATCH_SIZE), results):
            batch[i, :] = x.result()
    return batch


def print_time():
    timer = time.time()
    batch = form_batch()
    print("form_batch(): %s ms" % ((time.time() - timer) * 1000))

    timer = time.time()
    # preds = model.predict_on_batch(batch)
    preds = model.predict(batch)
    print("predict(): %s ms" % ((time.time() - timer) * 1000))

    timer = time.time()
    decode_predictions(preds)
    print("decode_predictions(): %s ms \n" % ((time.time() - timer) * 1000))


def previous_code():
    for n in range(10):
        batched_input = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
        tic = time.time()
        for i in range(BATCH_SIZE):
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            batched_input[i, :] = x
        batched_input = tf.constant(batched_input)
        print("batch_prepare(): %s ms" % ((time.time() - tic) * 1000))

        tic = time.time()
        preds = model.predict(batched_input)
        print("predict(): %s ms" % ((time.time() - tic) * 1000))

        tic = time.time()
        decode_predictions(preds)
        print("decode_predictions(): %s ms \n" % ((time.time() - tic) * 1000))


if __name__ == '__main__':
    for n in range(10):
        print_time()
