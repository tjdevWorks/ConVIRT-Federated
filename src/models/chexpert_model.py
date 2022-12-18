import torch

class CheXpert(torch.nn.Module):
    def __init__(self, image_model:torch.nn.Module, checkpoint_path:str='', freeze_backbone=True):
        super(CheXpert, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = image_model
        if len(checkpoint_path)!=0:
            ## Reloading Weights
            self.backbone.reload_model_weights(checkpoint_path)
        self.decoder = torch.nn.Sequential( torch.nn.Dropout(0.5),
                                            torch.nn.Linear(2048, 512),
                                            torch.nn.ReLU(),
                                            torch.nn.BatchNorm1d(512),
                                            torch.nn.Dropout(0.5),
                                            torch.nn.Linear(512, 5),
                                            torch.nn.Sigmoid(),
                                            )

    def forward(self, input_batch):
        with torch.set_grad_enabled(not self.freeze_backbone):
            encoding = self.backbone(input_batch)
        output = self.decoder(encoding)
        return output