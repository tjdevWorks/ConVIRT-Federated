from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import AutoTokenizer

from .chexpert_dataset import CheXpertDataSet, SquarePad


class CheXpertDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        train_fname: str,
        val_fname: str,
        test_fname: str,
        root_dir = '/',
        policy = 'ignore',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        # train_val_split: Tuple[float, float] = [0.95, 0.05],
        #cfg: dict = {},
    ):
        super(CheXpertDataModule,).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        ## Image transformations
        self.transform = transforms.Compose([
            #SquarePad(),
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.398,0.398,0.398), (0.327, 0.327, 0.327)),
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.prepare_data_per_node = False
        self.root_dir = root_dir
        self.policy = policy
        
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.val_fname = val_fname

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        return

    def setup(self, stage:str):
        """Load data
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        self.train_dataset = CheXpertDataSet(csv_name=self.train_fname,
                                transform = self.transform, 
                                policy = self.policy,
                                root_dir = self.root_dir,
                            )

        self.val_dataset = CheXpertDataSet(csv_name=self.val_fname,
                                transform = self.transform, 
                                policy = self.policy,
                                root_dir = self.root_dir,
                            )

        self.test_dataset = CheXpertDataSet(csv_name=self.test_fname,
                                transform = self.transform, 
                                policy = self.policy,
                                root_dir = self.root_dir,
                            )

    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "chexpert.yaml")
    _ = hydra.utils.instantiate(cfg)
