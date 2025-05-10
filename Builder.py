import os
import random
import cv2
import numpy as np

class DataBuilder:
    
    def get_distortions(self, image):
        """
        Returns 20 distorted versions of the image with random scaling between 80% and 120% of the original size
        """
        distortions = []
        for _ in range(20):
            x, y = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
            distorted = cv2.resize(image, (0, 0), fx=x, fy=y)
            distortions.append(distorted)
        return distortions

    def process_image_pipeline(self, image):
        """
        Returns the processed image with the following steps:
        1. Convert to grayscale
        2. Apply Canny edge detection --> Canny edges are used to detect the edges of the image
        3. Dilate the edges --> Dilating the edges helps to remove noise and improve the quality of the edges
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 100, 255)
        edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
        return edges
        
    def process_image(self, image_path : str):
        """
        Returns the processed image with the following steps:
        1. Read the image
        2. Process the image
        3. Find the contours
        4. For the contours, get the bounding boxes --> Bounding boxes are used to detect the objects in the image
        5. Return the cropped images --> Cropped images are extracted from the bounding boxes
        """
        image = cv2.imread(image_path)
        processed_image = self.process_image_pipeline(image)
                
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > 20 or h > 20):
                bounding_boxes.append((x, y, w, h))
                
        cropped_images = []
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            cropped_images.append(image[y:y+h, x:x+w])
        
        return cropped_images

    def build_data(self, screenshots_path : str, output_path : str) -> None:
        """
        Returns the processed image with the following steps:
        1. Read the image
        2. Process the image --> cropping the image to get significant objects
        3. For each cropped image, get distortions --> 20 disrtioned images with random scaling
        4. We save in a list a tuple of the cropped object image (that is present in the distorted image) and the distorted image
        5. Save in a path as a numpy array the data list
        """
        data = []
        for file in os.listdir(screenshots_path):
            if file.endswith(".png"):
                full_path = os.path.join(screenshots_path, file)
                cropped_images = self.process_image(full_path)
                for cropped_image in cropped_images:
                    distortions = self.get_distortions(cropped_image)
                    for distortion in distortions:
                        data.append((cropped_image, distortion))
                np.save(output_path, np.array(data, dtype=object))


if __name__ == "__main__":
    builder = DataBuilder()
    builder.build_data("data/screenshots", "data/results.npy")

