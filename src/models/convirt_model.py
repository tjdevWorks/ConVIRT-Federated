import torch

from .projection import Projector

class ConVIRT(torch.nn.Module):
    def __init__(self, image_model: torch.nn.Module, text_model:torch.nn.Module, proj_dim_size:int):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model

        self.image_projector = Projector(2048, 1024, proj_dim_size)
    
    def forward(self, input_batch):
        hv = self.image_model(input_batch['image'])
        v = self.image_projector(hv)
        u = self.text_model(input_batch['tokenized_data'])
        return v, u

class ConVIRTForImageClassfication(torch.nn.Module):
    def __init__(self):
        super(ConVIRTForImageClassfication).__init__()
    
    def forward(self):
        pass