from typing import Any, Dict, Optional, Tuple, List

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np

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
        policy = 'ignore',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_split: Tuple[float, float] = [0.95,0.05],
        normalize_mean_values: Tuple[float, float, float] = [0.485, 0.456, 0.406],
        normalize_std_values: Tuple[float, float, float] = [0.229, 0.224, 0.225],
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
            transforms.Normalize(normalize_mean_values, normalize_std_values)
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.prepare_data_per_node = False
        self.policy = policy
        
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.val_fname = val_fname

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        return

    def setup(self, stage:str, indices: List = [], debug_mode:bool=False):
        """Load data
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage=="fit":
            self.train_dataset = CheXpertDataSet(csv_name=self.train_fname,
                                transform = self.transform, 
                                policy = self.policy,
                            )
            ## Subset if indicies are there
            if len(indices)!=0:
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)

            total_samples = len(self.train_dataset)
            train_number_samples = int(self.hparams.train_val_split[0]*len(self.train_dataset))
            train_val_split_count = [train_number_samples, total_samples-train_number_samples]

            ## Uncomment Only while debugging
            if debug_mode:
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, np.arange(0, 100))
                train_val_split_count = [90, 10]

            self.train_dataset, self.val_dataset = random_split(
                    dataset=self.train_dataset,
                    lengths=train_val_split_count,
                    generator=torch.Generator().manual_seed(42),
            )
            print(f"Training Dataset Size: {len(self.train_dataset)}")
            print(f"Validation Dataset Size: {len(self.val_dataset)}")
        elif stage=="test":
            self.test_dataset = CheXpertDataSet(csv_name=self.val_fname,
                                transform = self.transform, 
                                policy = self.policy)

    def train_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True,
        )
    
    def val_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = False,
        )

    def test_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
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
