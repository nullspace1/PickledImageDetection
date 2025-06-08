from DataReceiver import DataReceiver
from typing import Callable
from threading import Lock
import numpy as np
import threading
from threading import Semaphore
import random

class SynchronizedQueue():
    def __init__(self):
        self.queue = []
        self.semaphore = Semaphore(0)
        
    def put(self, item):
        self.queue.append(item)
        self.semaphore.release()
            
    def get(self):
        self.semaphore.acquire()
        return self.queue.pop(0)
        

class DataProvider():
    def __init__(self, data_receiver : DataReceiver, reuse_probability : float = 0.3, max_reused_data : int = 100):
        self.data_receiver = data_receiver
        self.data_queue = SynchronizedQueue()
        self.reused_data = []
        self.reuse_probability = reuse_probability
        self.max_reused_data = max_reused_data
        
        
    def start_gathering(self):
        threading.Thread(target=self.gather_data).start()
            
    def gather_data(self):
        while True:
            self.data_receiver.listen()
            screenshot_shape = self.data_receiver.get_integers(3)
            template_shape = self.data_receiver.get_integers(3)
            screenshot_data = self.data_receiver.get_array(screenshot_shape)
            template_data = self.data_receiver.get_array(template_shape)
            rectangle = self.data_receiver.get_integers(4)
            self.data_queue.put((screenshot_data, template_data, rectangle))
        
    def get_next_data(self):
        if (random.random() < self.reuse_probability and len(self.reused_data) > 0):
            data = self.reused_data.pop(random.randint(0, len(self.reused_data) - 1))
        else:
            data = self.data_queue.get()
        if (random.random() < self.reuse_probability and len(self.reused_data) < self.max_reused_data):
            self.reused_data.append(data)
        return data
        
        