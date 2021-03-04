import cv2
import json
import requests
import numpy as np
from flask import Flask, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)


# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# data = json.dumps({
#     "instances": x.tolist()
# })
# headers = {"content-type": "application/json"}
# response = requests.post('http://localhost:8501/v1/models/resnet50:predict', data=data, headers=headers)
# response = json.loads(response.text)
# res = np.array(response["predictions"])
# print(decode_predictions(res))


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        image_file = request.files['file'].read()
        if image_file:
            # read images from URL post data
            image_array = np.fromstring(image_file, np.uint8)
            image_decode = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # preprocess image, resize it to 224x224
            img = cv2.resize(image_decode, dsize=(224, 224))
            img = np.expand_dims(img, axis=0)
            x = preprocess_input(img)

            # prepare the json data to send to the tf-serving server
            data = json.dumps({
                "instances": x.tolist()
            })
            headers = {"content-type": "application/json"}

            # send the request as an REST api
            response = requests.post('http://localhost:8501/v1/models/resnet50:predict', data=data, headers=headers)

            # get the classification result and return the processed result
            response = json.loads(response.text)
            res = np.array(response["predictions"])
            return decode_predictions(res)


if __name__ == '__main__':
    app.run('0.0.0.0')
