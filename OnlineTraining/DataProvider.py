import numpy as np
import threading
from threading import Semaphore
import random
import flask
from threading import Lock
import cv2
import os

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
    def __init__(self, host : str, port : int, data_variant_to_generate : int = 10, max_data : int = 500):
        self.data_queue = SynchronizedQueue()
        self.reused_data = []
        self.server = flask.Flask(__name__)
        self.host = host
        self.port = port
        self.counter = 0
        self.semaphore = Semaphore(0)
        self.data_variant_to_generate = data_variant_to_generate
        self.max_data = max_data
        self.data_folder = "data"
        self.lock = Lock()
        
        if (not os.path.exists(self.data_folder)):
            os.makedirs(self.data_folder)
        
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
            if not flask.request.content_type or 'multipart/form-data' not in flask.request.content_type:
                return flask.jsonify({'success': False, 'error': 'Request must be multipart/form-data'}), 415
                
            form_data = flask.request.form
            if not form_data:
                return flask.jsonify({'success': False, 'error': 'No form data received'}), 400
                
            try:

                rectangle_str = form_data.get('rectangle')
                
                if not all([rectangle_str]):
                    return flask.jsonify({'success': False, 'error': 'Missing required form fields'}), 400
                
                assert rectangle_str is not None
                
                rectangle = eval(rectangle_str)
                
                
                if 'screenshot' not in flask.request.files or 'template' not in flask.request.files:
                    return flask.jsonify({'success': False, 'error': 'Missing screenshot or template file'}), 400
                
                try:

                    screenshot_stream = flask.request.files['screenshot'].stream
                    template_stream = flask.request.files['template'].stream
                    
                    screenshot = cv2.imdecode(np.frombuffer(screenshot_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                    template = cv2.imdecode(np.frombuffer(template_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                  
                    rectangle = (rectangle[0], rectangle[1], rectangle[2], rectangle[3])

                    self.data_queue.put((screenshot, template, rectangle))
                    data = np.array([(
                        screenshot,
                        template,
                        np.array([rectangle[0], rectangle[1], rectangle[2], rectangle[3]])
                    )], dtype=[
                        ('screenshot', 'O'),
                        ('template', 'O'),
                        ('rectangle', 'O')
                    ])
                    self.lock.acquire()
                    np.save(f"{self.data_folder}/data_{self.counter}.npy", data)
                    self.counter += 1
                    self.lock.release()
                    self.semaphore.release()
                    return flask.jsonify({'success': True})
                except Exception as e:
                    return flask.jsonify({'success': False, 'error': f'Error loading images: {str(e)}'}), 400
            except Exception as e:
                return flask.jsonify({'success': False, 'error': str(e)}), 400

        
    def start_gathering(self):
        threading.Thread(target=self.gather_data_requests).start()
        threading.Thread(target=self.gather_data_processed).start()
            
    def gather_data_requests(self):
        print(f"Data provider running on {self.host}:{self.port}")
        self.server.run(host=self.host, port=self.port)
        
    def gather_data_processed(self):
        while True:
            
            self.semaphore.acquire()
            rnd_scr = random.randint(0, self.counter - 1)
            try:
                self.lock.acquire()
                data = np.load(f"{self.data_folder}/data_{rnd_scr}.npy", allow_pickle=True)[0]
                for i in range(self.data_variant_to_generate):
                    screenshot, template, rectangle = data['screenshot'], data['template'], data['rectangle']
                    random_x = random.uniform(0.9,1.1)
                    random_y = random.uniform(0.9,1.1)
                    new_template = cv2.resize(template, (int(template.shape[1] * random_x), int(template.shape[0] * random_y)))
                    self.data_queue.put((screenshot, new_template, rectangle))
                if (self.counter > self.max_data):
                    for i in range(self.max_data // 2):
                        os.remove(f"{self.data_folder}/data_{i}.npy")
                self.lock.release()
            except Exception as e:
                print(f"Error loading data: {e}")
                self.lock.release()
                continue
            
        
    def get_next_data(self):
        data = self.data_queue.get()
        return data
        
        