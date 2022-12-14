import os
import pickle

import cv2
import numpy as np
import pyvips
import torch
import csv
from PIL import Image
from torchvision import transforms

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.Pad(padding, 0, 'constant')(image)

class CheXpertDataSet(torch.utils.data.Dataset):
    def __init__(self,  csv_name:str, transform = None, policy = 'one', test_set=False):
        """
        csv_name: path to csv file
        transform: optional transform to be applied on a sample
        policy: to handle '-1' values in labelled data
        """
        super(CheXpertDataSet, self).__init__()

        csv_path = csv_name
        
        assert os.path.exists(csv_path), csv_path

        if policy == 'zero': 
            self.dict = {'1.0': 1, '': 0, '0.0': 0, "-1.0": 0}
        elif policy == 'one':
            self.dict = {'1.0': 1, '': 0, '0.0': 0, "-1.0": 1}
        else:
            self.dict = {'1.0': 1, '': 0, '0.0': 0}

        image_paths = []
        labels = []
        with open(csv_path, 'r', newline = '') as f:
            rows = csv.reader(f)
            # skipping the header
            next(rows, None)
            for row in rows:
                image_path = row[0]
                label_14 = row[5:] if not test_set else row[1:]
                label_5 = []
                ignore_row = False
                for i in [2, 5, 6, 8, 10]:
                    if policy == 'ignore' and label_14[i] == '-1.0':
                        ignore_row = True
                        break 
                    label_5.append(self.dict[label_14[i]])
                if policy == 'ignore' and ignore_row == True:
                    continue 
                image_paths.append(image_path)
                labels.append(label_5)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        label = self.labels[index]
        image_path = self.image_paths[index]
        
        # Read Image
        image = pyvips.Image.new_from_file(image_path, access="sequential")
        mem_img = image.write_to_memory()
        image = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)
        
        # Apply Transform
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': torch.FloatTensor(label)}

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    print("TODO: Need to write a basic test")