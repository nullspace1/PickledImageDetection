from os import listdir
import random as rd
import cv2
import os
import numpy as np

class Settings:
    def __init__(self, negative_sample_probability, max_crop_percentage, min_scale_x, max_scale_x, min_scale_y, max_scale_y, iterations):
        self.negative_sample_probability = negative_sample_probability
        self.max_crop_percentage = max_crop_percentage
        self.min_scale_x = min_scale_x
        self.max_scale_x = max_scale_x
        self.min_scale_y = min_scale_y
        self.max_scale_y = max_scale_y
        self.iterations = iterations
        

def __get_random_crop__(screenshot, settings):
    x,y = rd.randint(0, screenshot.shape[1] - 2), rd.randint(0, screenshot.shape[0]  - 2)
    width, height = rd.randint(1, int(settings.max_crop_percentage*(screenshot.shape[1] - x))), rd.randint(1, int(settings.max_crop_percentage* (screenshot.shape[0] - y)))
    rect = (x, y, width, height)
    crop = screenshot[y:y+height, x:x+width]
    scalex, scaley = rd.uniform(settings.min_scale_x, settings.max_scale_x), rd.uniform(settings.min_scale_y, settings.max_scale_y)
    crop_image = cv2.resize(crop, (0,0), fx=scalex, fy=scaley)
    return rect, crop_image

def __process_screenshot__(file):
    screenshot = cv2.imread(file)
    return screenshot

def __generate_training_data_pair__(file, files, settings):
    if (rd.uniform(0, 1) < settings.negative_sample_probability):
        crop_screenshot = __process_screenshot__(rd.choice(files))
        type = 0
    else:
        crop_screenshot = __process_screenshot__(file)
        type = 1
        
    crop_rect, crop_image = __get_random_crop__(crop_screenshot, settings)
    
    if type == 0:
        res = cv2.matchTemplate(crop_screenshot, crop_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.95:
            type = 1
        
    return crop_screenshot, crop_image, crop_rect, type


def get_training_data(screenshots_folder, settings, result_file):
    files = [os.path.join(screenshots_folder, f) for f in listdir(screenshots_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    training_data = []
    for i in range(settings.iterations):
        for file in files:
            screenshot, crop_image, crop_rect, type = __generate_training_data_pair__(file, files, settings)
            training_data.append((screenshot, crop_image, np.array(crop_rect), type))
    
    training_data = np.array(training_data, dtype=object) 
    np.save(result_file, training_data)
    return training_data

