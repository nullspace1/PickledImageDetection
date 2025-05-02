from os import listdir
import random as rd
import cv2
import os
import numpy as np

class Capturer:    
    def __init__(self,  negative_sample_probability : float, max_crop_percentage : float, min_scale_x : float,
                 max_scale_x : float, min_scale_y : float,
                 max_scale_y : float, iterations : int):
        self.negative_sample_probability = negative_sample_probability
        self.max_crop_percentage = max_crop_percentage
        self.min_scale_x = min_scale_x
        self.max_scale_x = max_scale_x
        self.min_scale_y = min_scale_y
        self.max_scale_y = max_scale_y
        self.iterations = iterations

    def __get_random_crop__(self,screenshot : np.ndarray) -> tuple[tuple[int,int,int,int], np.ndarray]:
        x,y = rd.randint(0, screenshot.shape[1] - 2), rd.randint(0, screenshot.shape[0]  - 2)
        width, height = rd.randint(1, int(self.max_crop_percentage*(screenshot.shape[1] - x))), rd.randint(1, int(self.max_crop_percentage* (screenshot.shape[0] - y)))
        rect = (x, y, width, height)
        crop_image = screenshot[y:y+height, x:x+width]
        return rect, crop_image


    def __generate_training_data_pair__(self,file : str, files : list[str]) -> tuple[np.ndarray, np.ndarray, tuple[int,int,int,int], int]:
        if (rd.uniform(0, 1) < self.negative_sample_probability):
            crop_screenshot = cv2.imread(rd.choice(files))
            type = 0
        else:
            crop_screenshot = cv2.imread(file)
            type = 1
            
        crop_rect, crop_image = self.__get_random_crop__(crop_screenshot)
        
        if type == 0:
            res = cv2.matchTemplate(crop_screenshot, crop_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.95:
                type = 1
                
        scalex, scaley = rd.uniform(self.min_scale_x,self.max_scale_x), rd.uniform(self.min_scale_y, self.max_scale_y)
        crop_image = cv2.resize(crop_image, (0,0), fx=scalex, fy=scaley)
            
        return crop_screenshot, crop_image, crop_rect, type


    def load_training_data(self,screenshots_folder : str, result_file : str) -> None:
        
        if os.path.exists(result_file):
            return
        
        files : list[str] = [os.path.join(screenshots_folder, f) for f in listdir(screenshots_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        training_data : np.ndarray = np.empty((0,4), dtype=object)
        
        for i in range(self.iterations):
            for file in files:
                screenshot, crop_image, crop_rect, type = self.__generate_training_data_pair__(file, files)
                training_data = np.append(training_data, np.array([(screenshot, crop_image, crop_rect, type)], dtype=object), axis=0)
                
        max_height, max_width = max([entry[1].shape[0] for entry in training_data]), max([entry[1].shape[1] for entry in training_data])
        
        for i in range(len(training_data)):
            sc, crop, rect, type = training_data[i]
            crop = cv2.copyMakeBorder(crop, 0, max_height - crop.shape[0], 0, max_width - crop.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
            training_data[i] = (sc, crop, rect, type)
        
        training_data = np.array(training_data, dtype=object) 
        np.save(result_file, training_data)


