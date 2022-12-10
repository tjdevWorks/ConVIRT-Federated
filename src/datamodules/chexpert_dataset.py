import os
import pickle

import cv2
import numpy as np
import pyvips
import torch
from torchvision import transforms

class CheXpertDataSet(torch.utils.data.Dataset):
    def __init__(self,  train:bool , config = None, transform = None, policy = 'ignore', root_dir = '/scratch/tm3647/public'):
        """
        train: bool var to determine csv path
        config: optional
        transform: optional transform to be applied on a sample
        policy: to handle '-1' values in labelled data
        root_dir: directory containing the CheXpert dataset
        """
        super(CheXpertDataSet, self).__init__()

        self.config = config
        self.root_dir = root_dir
        csv_name = 'CheXpert-v1.0/train.csv' if train else 'CheXpert-v1.0/valid.csv'
        csv_path = os.path.join(self.root_dir csv_name)

        assert os.path.exists(csv_path)

        if policy == 'zero': 
            self.dict = {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'}
        else if policy == 'one':
            self.dict = {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}
        else:
            self.dict = {'1.0': '1', '': '0', '0.0': '0'}

        image_paths = []
        labels = []
        with open(csv_path, 'r', newline = '') as f:
            rows = csv.reader(f)
            # skipping the header
            next(rows, None)
            for row in rows:
                image_path = row[0]
                label = row[5:]
                ignore_row = False
                for i in range(14):
                    if policy == 'ignore' and label[i] == '-1.0':
                        ignore_row = True
                        break 
                    label[i] = self.dict[label[i]]

                if policy == 'ignore' and ignore_row == True:
                    continue 
                image_paths.append(image_path)
                labels.append(label)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        label = self.labels[index]
        image_path = self.image_paths[index]
        image_path = os.path.join(self.root_dir, image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)   
        return {'image': image, 'label': torch.FloatTensor(label)}

    def __len__(self):
        return len(self.image_paths)

if __name__ == "__main__":
    print("TODO: Need to write a basic test")