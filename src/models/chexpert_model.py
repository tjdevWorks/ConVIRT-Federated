import torch

class CheXpert(torch.nn.Module):
    def __init__(self, image_model:torch.nn.Module, checkpoint_path:str, freeze_backbone=True):
        super(CheXpert, self).__init__()

        self.freeze_backbone = freeze_backbone
        # old_modules = [x for x in image_model.modules()]
        # backbone_out_features = old_modules[-1].out_features
        self.backbone = image_model
        # Reloading Weights
        self.backbone.reload_model_weights(checkpoint_path)
        self.decoder = torch.nn.Sequential( torch.nn.Dropout(0.2),
                                            torch.nn.Linear(2048, 14),
                                            torch.nn.Sigmoid(),
                                            )

    def forward(self, input_batch):
        with torch.set_grad_enabled(not self.freeze_backbone):
            encoding = self.backbone(input_batch)
        output = self.decoder(encoding)
        return output