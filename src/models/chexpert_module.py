from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, AUROC

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
        model: torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler,
        criterion:torch.nn.modules.loss,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])
        
        self.model = model
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_auc = AUROC(task="multilabel", num_labels=14, average="macro")
        self.val_auc = AUROC(task="multilabel", num_labels=14, average="macro")
        self.test_auc = AUROC(task="multilabel", num_labels=14, average="macro")

        self.train_mean_auc_score = MeanMetric()
        self.val_mean_auc_score = MeanMetric()
        self.test_mean_auc_score = MeanMetric()

        self.train_auc_per_class = AUROC(task="multilabel", num_labels=14, average=None)
        self.val_auc_per_class = AUROC(task="multilabel", num_labels=14, average=None)
        self.test_auc_per_class = AUROC(task="multilabel", num_labels=14, average=None)

        self.classes_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        pred = self.forward(batch['image'])
        loss = self.criterion(pred, batch['label'])
        return pred, loss

    def training_step(self, batch: Any, batch_idx: int):
        pred, loss = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        auc_score = self.train_auc(pred, batch['label'])
        self.train_mean_auc_score.update(auc_score)
        auc_scores_per_class = self.train_auc_per_class(pred, batch['label'])

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False,prog_bar=True)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "auc_per_class": auc_scores_per_class}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc_scores_batch_class = torch.stack(list(map(lambda x: x['auc_per_class'], outputs)))
        auc_scores_mean_class = torch.mean(auc_scores_batch_class, dim=0)
        for i, cls_name in enumerate(self.classes_names):
            self.log(f"train/{cls_name}/aucroc", auc_scores_mean_class[i],prog_bar=True)

        self.log("train/mean_aucroc", self.train_mean_auc_score.compute(), prog_bar=True)
        self.log("train/mean_loss", self.train_loss.compute(), prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        pred, loss = self.step(batch)
        
        self.val_loss(loss)
        auc_score = self.val_auc(pred, batch['label'])
        self.val_mean_auc_score.update(auc_score)
        auc_scores_per_class = self.val_auc_per_class(pred, batch['label'])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "auc_per_class": auc_scores_per_class}
    
    def validation_epoch_end(self, outputs: List[Any]):
        auc_scores_batch_class = torch.stack(list(map(lambda x: x['auc_per_class'], outputs)))
        auc_scores_mean_class = torch.mean(auc_scores_batch_class, dim=0)
        for i, cls_name in enumerate(self.classes_names):
            self.log(f"val/{cls_name}/aucroc", auc_scores_mean_class[i],prog_bar=True)

        self.log("val/mean_aucroc", self.val_mean_auc_score.compute(),prog_bar=True)
        self.log("val/mean_loss", self.val_loss.compute(), prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        pred, loss = self.step(batch)
        
        self.test_loss(loss)
        self.test_auc(pred, batch['label'])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/aucroc", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        trainable_model = self.model.decoder if self.model.freeze_backbone else self.model
        optimizer = self.hparams.optimizer(trainable_model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
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
