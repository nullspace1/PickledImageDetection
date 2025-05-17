import os
import numpy as np
import random
import cv2

class DataCreator:
    
    def __init__(self, screenshots_path, templates_path, batch_size = 10, samples = 10000, generated_data_path = "data/generated_data"):
        self.screenshots_path = screenshots_path
        self.templates_path = templates_path
        self.batch_size = batch_size
        self.samples = samples
        self.generated_data_path = generated_data_path
        
        os.makedirs(f"{self.generated_data_path}/screenshots", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/heatmaps", exist_ok=True)
        os.makedirs(f"{self.generated_data_path}/templates", exist_ok=True)
        
    def get_image(self, folder, path):
        # check if the image is a folder, if it is, get a random image from the folder
        if os.path.isdir(os.path.join(folder, path)):
            images = os.listdir(os.path.join(folder, path))
            return cv2.imread(os.path.join(folder, path, random.choice(images)))
        else:
            return cv2.imread(os.path.join(folder, path))
    
    def get_random_location(self, screenshot,templates, template):
        for n in range(100):
            x = random.randint(0, screenshot.shape[1] - template.shape[1])
            y = random.randint(0, screenshot.shape[0] - template.shape[0])
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
        screenshots_names = os.listdir(self.screenshots_path)
        templates_names = os.listdir(self.templates_path)
        data = []
        for _ in range(self.samples):
            screenshot = self.get_image(self.screenshots_path, random.choice(screenshots_names))
            templates = [(self.get_image(self.templates_path, random.choice(templates_names)), (-1, -1,-1,-1)) for _ in range(self.batch_size)]
            for i, (template,  box) in enumerate(templates):
                gen_box = self.get_random_location(screenshot,templates,template)
                if gen_box is not None:
                    
                    distortion_x, distortion_y = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
                    
                    box = (int(gen_box[0] * distortion_x), int(gen_box[1] * distortion_y), int(gen_box[2] * distortion_x), int(gen_box[3] * distortion_y))
                    distored_template = cv2.resize(template, (int(template.shape[1] * distortion_x), int(template.shape[0] * distortion_y)))
                    
                    screenshot = self.put_template_on_screenshot(screenshot,distored_template,box)
                    templates[i] = (template, box)
                            
                    heatmap = np.zeros((screenshot.shape[0], screenshot.shape[1]))
                    heatmap[box[1]:box[1] + template.shape[0], box[0]:box[0] + template.shape[1]] = 255
                    
          
                    cv2.imwrite(f"{self.generated_data_path}/screenshots/screenshot_{i}.jpg", screenshot)
                    cv2.imwrite(f"{self.generated_data_path}/heatmaps/heatmap_{i}.png", heatmap)
                    cv2.imwrite(f"{self.generated_data_path}/templates/template_{i}.jpg", template)
                    
                    data.append((f"{self.generated_data_path}/screenshots/screenshot_{i}.jpg",f"{self.generated_data_path}/heatmaps/heatmap_{i}.png", f"{self.generated_data_path}/templates/template_{i}.jpg"))
                    
        data = np.array(data, dtype=object)
        np.save(data_save_path, data)
                
if __name__ == "__main__":
    data_creator = DataCreator("data/screenshots", "data/templates",batch_size=1,samples=1)
    data_creator.create_data("data/training_data.npy")
    data = np.load("data/training_data.npy", allow_pickle=True)
    
    
    screenshot = data[0][0]
    
    print(screenshot)
