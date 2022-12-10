import torch

class CheXpert(torch.nn.Module):
    def __init__(self, image_model:torch.nn.Module, freeze_backbone=True):
        super(CheXpert, self).__init__()

        self.freeze_backbone = freeze_backbone
        old_modules = [x for x in image_model.modules()]
        backbone_out_features = old_modules[-1].in_features
        self.backbone = nn.Sequential(*old_modules[1:-1])
        self.decoder = nn.Sequential(nn.Linear(backbone_out_features, 14),
                                            nn.Sigmoid(),
                                            )
    
    def forward(self, input_batch):
        with torch.set_grad_enabled(not self.freeze_backbone):
            encoding = self.backbone(input_batch)
        output = self.decoder(encoding)
        return output