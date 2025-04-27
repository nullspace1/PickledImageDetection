import capturer as dc
import numpy as np
import cv2

settings = dc.Settings(
    negative_sample_probability=0.1,
    max_crop_percentage=0.8,
    min_scale_x=0.8,
    max_scale_x=1.5,
    min_scale_y=0.8,
    max_scale_y=1.5,
    iterations=10
)

data = dc.get_training_data( './data/screenshots',settings,'./data/data.npy')


for d in data:
    (screenshot, crop_image, crop_rect) = d
    print(screenshot[0][0][0])