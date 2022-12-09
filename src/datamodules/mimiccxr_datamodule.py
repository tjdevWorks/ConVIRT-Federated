from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import AutoTokenizer

from .mimiccxr_dataset import MIMICCXRDataset

class MIMICCXRDataModule(LightningDataModule):
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
        mimic_cxr_dataset_file:str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        text_model_name:str = '',
        train_val_split: Tuple[float, float] = [0.95,0.05],
        #cfg: dict = {},
    ):
        super(MIMICCXRDataModule,).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        ## Image transformations
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, ratio=[0.6, 1.0]),
            transforms.RandomAffine(degrees=[-20,20], translate=(0.1,0.1), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
            #transforms.GaussianBlur(G) ## Not implemented due to no info on kernel size in the paper
            #transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize((0.398,0.398,0.398), (0.327, 0.327, 0.327))
        ])
        
        ## Transforms
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.prepare_data_per_node = False

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
        dataset = MIMICCXRDataset(self.hparams.mimic_cxr_dataset_file, transforms=self.transforms, tokenizer=self.tokenizer)
        total_samples = len(dataset)
        train_number_samples = int(self.hparams.train_val_split[0]*len(dataset))
        self.hparams.train_val_split = [train_number_samples, total_samples-train_number_samples]
        self.train_dataset, self.val_dataset = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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
    
    def collate_and_tokenize(self, batch):
        input_data = {}
        
        images = torch.cat(list(map(lambda x: torch.unsqueeze(x['image'], 0), batch)))
        texts = list(map(lambda x: x['text'], batch))
        
        input_data = self.tokenizer.batch_encode_plus(texts, max_length=128, padding=True, truncation=True, return_tensors="pt")
        
        keys = list(input_data.keys())
        input_data['tokenized_text'] = {}
        
        for k in list(keys):
            input_data['tokenized_text'][k]=input_data.pop(k)
        
        input_data['images'] = images
        input_data['texts'] = texts
        
        return input_data

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mimic_cxr.yaml")
    _ = hydra.utils.instantiate(cfg)
