from typing import Dict

import torch
from transformers import AutoModel

from .projection import Projector

class TextEncoder(torch.nn.Module):
    def __init__(self, name: str, proj_dim_size:int):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(name)
        
        # Freeze the first 6 layers of BERT
        for i, p in enumerate(self.model.named_parameters()):
            if i==101:
                break
            p[1].requires_grad = False
        
        self.projector = Projector(768, 768, proj_dim_size)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, texts_tokenized_data: Dict):
        bert_model_output = self.model(**texts_tokenized_data)
        
        ## Apply mean_pooling
        hu = self.mean_pooling(bert_model_output, texts_tokenized_data['attention_mask'])
        
        ## Projection
        u = self.projector(hu)
        
        return u