import os
import pickle

import cv2
import numpy as np
import pyvips
import torch
from torchvision import transforms

class MIMICCXRDataset(torch.utils.data.Dataset):
    def __init__(self,  df_file:str , config=None, transforms=None, tokenizer=None):
        super(MIMICCXRDataset, self).__init__()
        
        self.config = config
        assert os.path.exists(df_file) and os.path.splitext(df_file)[1].lower()==".pkl", "Check file path exists and has the extension .pkl"
        
        with open(df_file, 'rb') as f:
            self.df = pickle.load(f)
        
        self.transforms = transforms
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        record = self.df.iloc[int(idx)]
        
        image = pyvips.Image.new_from_file(record['image_fname'], access="sequential")
        mem_img = image.write_to_memory()
        image = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width)
        image = transforms.ToTensor()(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        
        findings = record['findings_tokenized_sentences']
        impressions = record['impressions_tokenized_sentences']
        
        find_impres = findings + impressions
        
        assert len(find_impres)!=0, f"Issue findings/impression of {record['patient_folder']}/{record['patient_id']}/{record['study_id']}"
        
        text = np.random.choice(find_impres)
        
        if self.transforms:
            image = self.transforms(image)
        
        if self.tokenizer:
            tokenized_input_data = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_input_data = {k: v.squeeze() for k, v in tokenized_input_data.items()}
        
        return {'image':image, 'text': text, 'tokenized_data':tokenized_input_data}

if __name__ == "__main__":
    print("TODO: Need to write a basic test")