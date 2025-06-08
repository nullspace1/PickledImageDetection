import socket
import struct
import numpy as np

class DataReceiver():
    
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr) and not name.startswith('__'):
            def wrapper(*args, **kwargs):
                print(f"Calling method: {name}")
                return attr(*args, **kwargs)
            return wrapper
        return attr
        
    def __init__(self, port : int, host : str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.s = None
        self.conn = None
        self.data = b''
        
    def listen(self):
        
        if self.s is None:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.s.bind((self.host, self.port))
            self.s.listen(1)
        
        
        print("Waiting for connection...")
        self.conn, addr = self.s.accept()
        print(f"Connected to {addr}")
    
    def get_integers(self, n : int):
        if self.conn is None:
            raise ConnectionError("Not connected to a client")
        data = b''
        while len(data) < 4 * n:
            packet = self.conn.recv(4 * n - len(data))
            if not packet:
                raise ConnectionError("Connection closed by client")
            data += packet
        result = struct.unpack(f'>{n}I', data)
        print(f"Received integers: {result}")
        return result
    
    def get_array(self, shape : tuple):
        if self.conn is None:
            raise ConnectionError("Not connected to a client")
        total_size = np.prod(shape)
        data = b''
        while len(data) < total_size:
            packet = self.conn.recv(min(4096, (int) (total_size - len(data))))
            if not packet:
                raise ConnectionError("Connection closed by client")
            data += packet
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)
    
    def close(self):
        if self.conn:
            self.conn.close()
        if self.s:
            self.s.close()
        self.conn = None
        self.s = None