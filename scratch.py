import time
import threading
from gevent.pywsgi import WSGIServer
from multiprocessing.dummy import Pool as ThreadPool
from flask import Flask, request
from queue import Empty, Queue

app = Flask(__name__)
req_queue = Queue()


def print_num(thread):
    for i in range(10):
        print(thread + ": " + str(i))


@app.route('/')
def index():
    return "hello world"


@app.route('/batch')
def batch_processing():
    image = request.args.get("image")
    req = {"input": image, "time": time.time()}
    # put request into queue
    req_queue.put(req)
    # wait for the request to be processed
    while "output" not in req:
        time.sleep(0.01)
    # if the request has "output", return the value
    return req["output"]


def predict(image):
    image['output'] = image['input'] + " predicted"


def handle_requests():
    while True:
        batch = []
        while not (len(batch) > 10 or (len(batch) > 0 and time.time() - batch[0]['time'] > 0.5)):
            try:
                batch.append(req_queue.get(timeout=0.01))
            except Empty:
                continue
        pool = ThreadPool()
        pool.map(predict, batch)


threading.Thread(target=handle_requests).start()

# l = ["a", "b", "c"]
# pool = ThreadPool()
# pool.map(print_num, l)

if __name__ == '__main__':
    # app.run('0.0.0.0')
    WSGIServer(('0.0.0.0', 5000), app).serve_forever()
