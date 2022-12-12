from typing import Optional, Any
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

class ImageEncoder(torch.nn.Module):
    def __init__(self, name: str, frozen: bool=False, weights: Optional[Any]=None, return_interm_layers: bool=False):
        super().__init__()
        backbone = getattr(torchvision.models, name)(weights=weights)

        ## Setting layers to be non-trainable
        # for name, parameter in backbone.named_parameters():
        #     if frozen or "layer2" not in name and "layer3" not in name and "layer4" not in name:
        #         parameter.requires_grad_(False)
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3",  "avgpool": "hv"}
        else:
            return_layers = {"layer4": "0",  "avgpool": "hv"}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def reload_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    def forward(self, images):
        out = self.body(images)
        ## Get the output of the avgpool (B, 2048, 1, 1)
        return out["hv"].squeeze()