import os
import numpy as np
import random
import cv2
import torch
from tqdm import tqdm

class DataCreator:
    
    def __init__(self, screenshots_path, templates_path,generated_data_path, templates_per_screenshot = 10, samples = 10000):
        self.screenshots_path = screenshots_path
        self.templates_path = templates_path
        self.templates_per_screenshot = templates_per_screenshot
        self.samples = samples
        self.generated_data_path = generated_data_path
        self.MAX_SIZE = 50 * 1024 * 1024
        self.MAX_DISTORTION = (1,1)
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
      
    def get_random_location(self, screenshot,templates, template):
        if screenshot.shape[0] < template.shape[0] or screenshot.shape[1] < template.shape[1]:
            return None
        for n in range(100):
            x = random.randint(0, screenshot.shape[1] - int(template.shape[1] * self.MAX_DISTORTION[1]))
            y = random.randint(0, screenshot.shape[0] - int(template.shape[0] * self.MAX_DISTORTION[1]))
            if self.is_valid_location(templates, (x,y)):
                return (x, y, template.shape[1], template.shape[0])
        return None
    
    def is_valid_location(self, templates, box):
        for (template, b) in templates:
            if b != (-1, -1, -1, -1):
                if b[0] <= box[0] <= b[0] + template.shape[1] and b[1] <= box[1] <= b[1] + template.shape[0]:
                    return False
        return True
    
    def get_templates(self, templates_names):
        templates = []
        for _ in range(self.templates_per_screenshot):
            template = self.get_image(random.choice(templates_names))
            templates.append((template, (-1, -1,-1,-1)))
        return templates
    
    def assign_boxes(self, templates, screenshot):
        for i, (template, box) in enumerate(templates):
            box = self.get_random_location(screenshot, templates, template)
            if box is not None:
                templates[i] = (template, box)
                
    def distort_templates(self, templates):
        for i, (template, box) in enumerate(templates):
            distortion_x, distortion_y = random.uniform(self.MAX_DISTORTION[0], self.MAX_DISTORTION[1]), random.uniform(self.MAX_DISTORTION[0], self.MAX_DISTORTION[1])
            distored_template = cv2.resize(template, (int(template.shape[1] * distortion_x), int(template.shape[0] * distortion_y)))
            templates[i] = (distored_template, box)
            
    def put_templates_on_screenshot(self, screenshot, templates):
        for template, box in templates:
            if box != (-1, -1, -1, -1):
             screenshot[box[1]:box[1] + template.shape[0], box[0]:box[0] + template.shape[1]] = template
        return screenshot
    
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
        templates_names = self.scan_files(self.templates_path)
        
        data = []
        for _ in tqdm(range(self.samples), desc="Generating data"):
            
            screenshot = self.get_image(random.choice(screenshots_names))
            templates = self.get_templates(templates_names)
            
            self.assign_boxes(templates, screenshot)
            self.distort_templates(templates)
            self.put_templates_on_screenshot(screenshot, templates)
            
            template, box = random.choice([(template, box) for template, box in templates if box != (-1, -1, -1, -1)])
            heatmap = self.create_heatmap(screenshot, template, box)
            
            path_screenshot, path_heatmap, path_template = self.save_data(screenshot, heatmap, template)
            
            data.append((path_screenshot,path_heatmap, path_template))
                    
        data = np.array(data, dtype=object)
        np.save(data_save_path, data)
            