import numpy as np
import threading
from threading import Semaphore
import random
import flask
from threading import Lock
import cv2

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
            if not flask.request.content_type or 'multipart/form-data' not in flask.request.content_type:
                return flask.jsonify({'success': False, 'error': 'Request must be multipart/form-data'}), 415
                
            form_data = flask.request.form
            if not form_data:
                return flask.jsonify({'success': False, 'error': 'No form data received'}), 400
                
            try:
                screenshot_sizes_str = form_data.get('screenshot_sizes')
                template_sizes_str = form_data.get('template_sizes')
                rectangle_str = form_data.get('rectangle')
                
                if not all([screenshot_sizes_str, template_sizes_str, rectangle_str]):
                    return flask.jsonify({'success': False, 'error': 'Missing required form fields'}), 400
                
                assert screenshot_sizes_str is not None
                assert template_sizes_str is not None
                assert rectangle_str is not None
                
                screenshot_sizes = eval(screenshot_sizes_str)
                template_sizes = eval(template_sizes_str)
                rectangle = eval(rectangle_str)
                
                screenshot_shape = (screenshot_sizes[0], screenshot_sizes[1], screenshot_sizes[2])
                template_shape = (template_sizes[0], template_sizes[1], template_sizes[2])
                
                if 'screenshot' not in flask.request.files or 'template' not in flask.request.files:
                    return flask.jsonify({'success': False, 'error': 'Missing screenshot or template file'}), 400
                
                try:

                    screenshot_stream = flask.request.files['screenshot'].stream
                    template_stream = flask.request.files['template'].stream
                    
                    screenshot = cv2.imdecode(np.frombuffer(screenshot_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                    template = cv2.imdecode(np.frombuffer(template_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                    
                    if screenshot.shape != screenshot_shape:
                        screenshot = cv2.resize(screenshot, (screenshot_shape[1], screenshot_shape[0]))
                    if template.shape != template_shape:
                        template = cv2.resize(template, (template_shape[1], template_shape[0]))
                        
                    rectangle = (rectangle[0], rectangle[1], rectangle[2], rectangle[3])
                    
                    self.data_queue.put((screenshot, template, rectangle))
                    return flask.jsonify({'success': True})
                except Exception as e:
                    return flask.jsonify({'success': False, 'error': f'Error loading images: {str(e)}'}), 400
            except Exception as e:
                return flask.jsonify({'success': False, 'error': str(e)}), 400

        
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
        
        