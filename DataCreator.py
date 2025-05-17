import os
import numpy as np
import random
import cv2
from tqdm import tqdm

class DataCreator:
    
    def __init__(self, screenshots_path, templates_path, templates_per_screenshot = 10, samples = 10000, generated_data_path = "data/generated_data"):
        self.screenshots_path = screenshots_path
        self.templates_path = templates_path
        self.templates_per_screenshot = templates_per_screenshot
        self.samples = samples
        self.generated_data_path = generated_data_path
        
        os.makedirs(f"{self.generated_data_path}/screenshots", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/heatmaps", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/templates", exist_ok=True)
        
    def get_image(self, folder, path):
        try:
            if os.path.isdir(os.path.join(folder, path)) and any(f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) for f in os.listdir(os.path.join(folder, path))):
                images = [f for f in os.listdir(os.path.join(folder, path)) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]  
                if not images:
                    raise ValueError(f"No valid images found in {os.path.join(folder, path)}")
                img = cv2.imread(os.path.join(folder, path, random.choice(images)))
            else:
                img = cv2.imread(os.path.join(folder, path))
            
            if img is None:
                raise ValueError(f"Failed to load image from {os.path.join(folder, path)}")
                
            return img
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            # Return a default image or raise an error
            raise ValueError(f"Could not load image from {os.path.join(folder, path)}: {str(e)}")
    
    def get_random_location(self, screenshot,templates, template):
        for n in range(100):
            x = random.randint(0, screenshot.shape[1] - int(template.shape[1] * 1.2))
            y = random.randint(0, screenshot.shape[0] - int(template.shape[0] * 1.2))
            if self.is_valid_location(templates, (x,y)):
                return (x, y, template.shape[1], template.shape[0])
        return None
    
    def is_valid_location(self, templates, box):
        for (template, b) in templates:
            if b != (-1, -1, -1, -1):
                if b[0] <= box[0] <= b[0] + template.shape[1] and b[1] <= box[1] <= b[1] + template.shape[0]:
                    return False
        return True
    
    def put_template_on_screenshot(self, screenshot, template, box):
        screenshot[box[1]:box[1] + template.shape[0], box[0]:box[0] + template.shape[1]] = template
        return screenshot

    def create_data(self, data_save_path):
        ## add a loading bar
        screenshots_names = [f for f in os.listdir(self.screenshots_path) if not f.startswith('.')]
        templates_names = [f for f in os.listdir(self.templates_path) if not f.startswith('.')]
        data = []
        for _ in tqdm(range(self.samples), desc="Generating data"):
            screenshot = self.get_image(self.screenshots_path, random.choice(screenshots_names))
            templates = [(self.get_image(self.templates_path, random.choice(templates_names)), (-1, -1,-1,-1)) for _ in range(self.templates_per_screenshot)]
            for i, (template,  box) in enumerate(templates):
                gen_box = self.get_random_location(screenshot,templates,template)
                if gen_box is not None:
                    
                    distortion_x, distortion_y = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
                    
                    distored_template = cv2.resize(template, (int(template.shape[1] * distortion_x), int(template.shape[0] * distortion_y)))
                    box = (gen_box[0], gen_box[1], distored_template.shape[1], distored_template.shape[0])
                    
                    screenshot = self.put_template_on_screenshot(screenshot,distored_template,box)
                    templates[i] = (template, box)
                            
                    heatmap = np.zeros((screenshot.shape[0], screenshot.shape[1]))
                    heatmap[box[1]:box[1] + template.shape[0], box[0]:box[0] + template.shape[1]] = 255
                    
                    path_screenshot = f"{self.generated_data_path}/screenshots/screenshot_{random.randint(0, 1000000)}.jpg"
                    path_heatmap = f"{self.generated_data_path}/heatmaps/heatmap_{random.randint(0, 1000000)}.png"
                    path_template = f"{self.generated_data_path}/templates/template_{random.randint(0, 1000000)}.jpg"
          
                    cv2.imwrite(path_screenshot, screenshot)
                    cv2.imwrite(path_heatmap, heatmap)
                    cv2.imwrite(path_template, template)
                    
                    data.append((path_screenshot,path_heatmap, path_template))
                    
        data = np.array(data, dtype=object)
        np.save(data_save_path, data)
                
if __name__ == "__main__":
    data_creator = DataCreator("data/screenshots", "data/templates",templates_per_screenshot=5,samples=1)
    data_creator.create_data("data/training_data.npy")
    data = np.load("data/training_data.npy", allow_pickle=True)
    
    
    screenshot = data[0][0]
    
    print(screenshot)
