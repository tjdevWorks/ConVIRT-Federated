from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from .convirt_module import ConVIRTLitModule
from .image_encoder import ImageEncoder
from .chexpert_model import CheXpert

class CheXpertLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        checkpoint_path:str,
        # net: torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler,
        criterion:torch.nn.modules.loss,
        freeze_backbone = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["criterion"])
        self.freeze_backbone = freeze_backbone

        image_model = ImageEncoder('resnet50')#ConVIRTLitModule.load_from_checkpoint(checkpoint_path).net.image_model
        self.model = CheXpert(image_model, freeze_backbone)

        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        pred = self.forward(batch['image'])
        loss = self.criterion(pred, batch['label'])
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # update and log metrics
        self.train_loss(loss)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        trainable_model = self.model.decoder if self.freeze_backbone else self.model
        optimizer = self.hparams.optimizer(trainable_model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 5000,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "chexpert_finetune_model.yaml")
    _ = hydra.utils.instantiate(cfg)
