import numpy as np
import threading
from threading import Semaphore
import random
import flask
from threading import Lock

class SynchronizedQueue():
    def __init__(self):
        self.queue = []
        self.semaphore = Semaphore(0)
        self.lock = Lock()
        
    def put(self, item):
        with self.lock:
            self.queue.append(item)
            self.semaphore.release()
            
    def get(self):
        self.semaphore.acquire()
        return self.queue.pop(0)
        

class DataProvider():
    def __init__(self, host : str, port : int, reuse_probability : float = 0.3, max_reused_data : int = 100):
        self.data_queue = SynchronizedQueue()
        self.reused_data = []
        self.reuse_probability = reuse_probability
        self.max_reused_data = max_reused_data
        self.server = flask.Flask(__name__)
        self.host = host
        self.port = port
        
        ## POST FORMAT
        ## {
        ##     "screenshot_sizes": [int, int, int],
        ##     "template_sizes": [int, int, int],
        ##     "rectangle": [int, int, int, int]
        ##     "screenshot": base64,
        ##     "template": base64
        ## }
        ## Returns: {"success": True}
        
        @self.server.route('/', methods=['POST'])
        def data():
            data = flask.request.get_json()
            screenshot_sizes = data['screenshot_sizes']
            template_sizes = data['template_sizes']
            rectangle = data['rectangle']
            
            screenshot_shape = (screenshot_sizes[0], screenshot_sizes[1], screenshot_sizes[2])
            template_shape = (template_sizes[0], template_sizes[1], template_sizes[2])
            
            screenshot = np.load(flask.request.files['screenshot'].stream).reshape(screenshot_shape)
            template = np.load(flask.request.files['template'].stream).reshape(template_shape)
            rectangle = (rectangle[0], rectangle[1], rectangle[2], rectangle[3])
            
            self.data_queue.put((screenshot, template, rectangle))
            return flask.jsonify({'success': True})

        
    def start_gathering(self):
        threading.Thread(target=self.gather_data).start()
            
    def gather_data(self):
        print(f"Data provider running on {self.host}:{self.port}")
        self.server.run(host=self.host, port=self.port)
        
    def get_next_data(self):
        if (random.random() < self.reuse_probability and len(self.reused_data) > 0):
            data = self.reused_data.pop(random.randint(0, len(self.reused_data) - 1))
        else:
            data = self.data_queue.get()
        if (random.random() < self.reuse_probability):
            self.reused_data.append(data)
        if (len(self.reused_data) > self.max_reused_data):
            self.reused_data.pop(0)
        return data
        
        