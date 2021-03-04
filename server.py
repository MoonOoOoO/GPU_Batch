import threading
import time
from flask import Flask, request
from queue import Empty, Queue
import multiprocessing
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
requests_queue = Queue()

BATCH_SIZE = 10
BATCH_TIMEOUT = 0.1
CHECK_INTERVAL = 0.01


def print_num(thread):
    for i in range(10):
        print(thread + ": " + str(i))


def generate_single_output(string):
    return string + ": " + str(time.strftime("%H:%M:%S", time.localtime(time.time()))) + ", batch size = " + str(
        BATCH_SIZE)


def generate_batch_output(req):
    req['output'] = generate_single_output(req['input'])


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) > BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                # next_req = requests_queue.get(timeout=CHECK_INTERVAL)
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                pass
            # else:
            #     requests_batch.append(next_req)

        for r in requests_batch:
            p = multiprocessing.Process(target=generate_batch_output, args=(r,))
            p.start()
            p.join()
        # for r in requests_batch:
        #     # r['output'] = r['input'] + ": " + str(time.strftime("%H:%M:%S", time.localtime(time.time())))
        #     r['output'] = generate_single_output(r['input'])


@app.route('/')
def hello_world():
    time.sleep(0.1)
    return 'Hello, World!'


@app.route('/nob')
def non_batch_processing():
    image = request.args.get("image")
    return generate_single_output(image)


@app.route('/batch')
def batch_processing():
    image = request.args.get("image")
    img_req = {"input": image, "time": time.time()}
    requests_queue.put(img_req)
    while "output" not in img_req:
        time.sleep(CHECK_INTERVAL)

    return img_req['output']


if __name__ == '__main__':
    t = threading.Thread(target=handle_requests_by_batch)
    t.daemon = True
    t.start()
    app.run('0.0.0.0')
    # WSGIServer(('0.0.0.0', 5000), app).serve_forever()
