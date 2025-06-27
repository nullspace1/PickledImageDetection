import os
import numpy as np
import random
import cv2
import torch
from tqdm import tqdm

class DataCreator:
    
    def __init__(self, screenshots_path,generated_data_path, templates_per_screenshot = 10, samples = 10000):
        self.screenshots_path = screenshots_path
        self.templates_per_screenshot = templates_per_screenshot
        self.samples = samples
        self.generated_data_path = generated_data_path
        self.MAX_SIZE = 50 * 1024 * 1024
        
        os.makedirs(f"{self.generated_data_path}/screenshots", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/heatmaps", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/templates", exist_ok=True)
        
    def scan_files(self, folder):
        file_list = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path) and os.path.getsize(full_path) < self.MAX_SIZE:
                        file_list.append(full_path)
        return file_list
        
    def get_image(self, path):
        return cv2.imread(path)
      
    def get_templates(self, screenshot):
        templates = []
        screenshot_edges = cv2.Canny(screenshot, 100, 200)
        contours, _ = cv2.findContours(screenshot_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _ in range(min(self.templates_per_screenshot, len(contours))):
            contour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(contour)
            template = screenshot[y:y+h, x:x+w]
            templates.append((template, (x, y, w, h)))
        return templates
    
    def create_heatmap(self, screenshot, template, box):
        heatmap = np.zeros((screenshot.shape[0], screenshot.shape[1]), dtype=np.uint8)
        heatmap[box[1]:box[1] + template.shape[0], box[0]:box[0] + template.shape[1]] = 255
        return heatmap[:,:,np.newaxis]
    
    def save_data(self, screenshot, heatmap, template): 
        path_screenshot = f"{self.generated_data_path}/screenshots/screenshot_{random.randint(0, 1000000)}"
        path_heatmap = f"{self.generated_data_path}/heatmaps/heatmap_{random.randint(0, 1000000)}"
        path_template = f"{self.generated_data_path}/templates/template_{random.randint(0, 1000000)}"
        np.save(path_screenshot, screenshot)
        np.save(path_heatmap, heatmap)
        np.save(path_template, template)
        return path_screenshot + ".npy", path_heatmap + ".npy", path_template + ".npy"
 
    def create_data(self, data_save_path):
        screenshots_names = self.scan_files(self.screenshots_path)
        
        data = []
        for _ in tqdm(range(self.samples), desc="Generating data"):
            
            screenshot = self.get_image(random.choice(screenshots_names))
            templates = self.get_templates(screenshot)
            
            for template, box in templates:
                heatmap = self.create_heatmap(screenshot, template, box)
            
            path_screenshot, path_heatmap, path_template = self.save_data(screenshot, heatmap, template)
            data.append((path_screenshot,path_heatmap, path_template))
                    
        data = np.array(data, dtype=object)
        np.save(data_save_path, data)
            