from DataReceiver import DataReceiver
from typing import Callable
from threading import Lock
import numpy as np
import threading
from threading import Semaphore

class SynchronizedQueue():
    def __init__(self, maxsize=100):
        self.queue = []
        self.semaphore = Semaphore(0)
        self.maxsize = maxsize
        
    def put(self, item):
        if len(self.queue) >= self.maxsize:
            self.queue.pop(0)  # Remove oldest item if queue is full
        self.queue.append(item)
        self.semaphore.release()
            
    def get(self):
        self.semaphore.acquire()
        return self.queue.pop(0)
        

class DataProvider():
    def __init__(self, data_receiver : DataReceiver):
        self.data_receiver = data_receiver
        self.data_queue = SynchronizedQueue()
        
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
        return self.data_queue.get()
        
        
        