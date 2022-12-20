from typing import Dict, List, Callable, Optional, Tuple
from collections import OrderedDict
import os
from path import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from omegaconf import OmegaConf
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config_filename: str, indices: List, hydra_runtime_log_dir, batch_size_override:int=None):
        ## TODO: Think how can you have the clean up function setup like in utils.task_wrapper
        self.cid = cid

        ## Load the hydra config
        cfg = self.load_client_config(config_filename)

        ## Reset certain cfg values
        cfg = self.reset_specific_cfg_values(cfg, hydra_runtime_log_dir)

        if batch_size_override is not None:
            cfg['datamodule']['batch_size'] = batch_size_override

        ## set seed for random number generators in pytorch, numpy and python.random
        if cfg.get("seed"):
            pl.seed_everything(cfg.seed, workers=True)

        ## Create client directory if it doesn't exist
        if not os.path.exists(f"{hydra_runtime_log_dir}/{cid}"):
            os.mkdir(f"{hydra_runtime_log_dir}/{cid}")

        ## Instantiate model
        self.model = hydra.utils.instantiate(cfg.model)

        ## Create Data Module and call setup
        self.datamodule = hydra.utils.instantiate(cfg.datamodule)
        self.datamodule.setup("fit", indices=indices, debug_mode=cfg.datamodule_debug_mode)

        #print(f"Created all objects for : {self.cid}")
        
        self.cfg = cfg

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_client_config(self, config_fname):
        #print(f"Client: {self.cid}")
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base="1.2", config_path="../configs/")
        cfg = compose(config_name=config_fname, return_hydra_config=True)
        HydraConfig().cfg = cfg
        OmegaConf.resolve(cfg)
        return cfg

    def reset_specific_cfg_values(self, cfg, runtime_dir):
        ## Paths output_dir
        cfg['paths']['output_dir'] = str(Path(runtime_dir) / f'{self.cid}')
        ## logger csv save dir
        cfg['logger']['csv']['save_dir'] = str(Path(runtime_dir) / f'{self.cid}')
        ## callbacks model checkpoint
        cfg['callbacks']['model_checkpoint']['dirpath'] = str(Path(runtime_dir) / f'{self.cid}')
        ## Trainer default_root_dir
        cfg['trainer']['default_root_dir'] = str(Path(runtime_dir) / f'{self.cid}')
        return cfg
    
    def set_server_round_dir(self, server_round):
        ## logger csv save dir
        self.cfg['logger']['csv']['save_dir'] = str(Path(self.cfg['logger']['csv']['save_dir']) / f"{server_round}")
        ## callbacks model checkpoint
        self.cfg['callbacks']['model_checkpoint']['dirpath'] = str(Path(self.cfg['callbacks']['model_checkpoint']['dirpath']) / f"checkpoints/")
        #str(Path(self.cfg['callbacks']['model_checkpoint']['dirpath']) / f"{server_round}/checkpoints/")
        

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        ## Set the directories
        self.set_server_round_dir(config["server_round"])

        self.callbacks: List[Callback] = utils.instantiate_callbacks(self.cfg.callbacks)

        ## Delete the checkpoints dirpath last.ckpt
        if os.path.exists(self.callbacks[0].dirpath+'/last.ckpt'):
            os.remove(self.callbacks[0].dirpath+'/last.ckpt')

        self.logger: List[LightningLoggerBase] = utils.instantiate_loggers(self.cfg.get("logger"))

        ## Create Trainer Object
        self.trainer: Trainer = hydra.utils.instantiate(self.cfg.trainer, callbacks=self.callbacks, logger=self.logger)

        ## For model checkpoint change output directory
        if type(self.callbacks[0])==pl.callbacks.ModelCheckpoint:
            os.makedirs(self.callbacks[0].dirpath, exist_ok=True)

        ## For csv logging change save directory
        if type(self.logger[0])==pl.loggers.csv_logs.CSVLogger:
            os.makedirs(self.logger[0].save_dir, exist_ok=True)

        set_params(self.model, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])#int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        
        ## Create data loaders
        train_dataloader = self.datamodule.train_dataloader(num_workers=num_workers)
        val_dataloader = self.datamodule.val_dataloader(num_workers=num_workers)
        
        self.trainer.fit(model=self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Return local model and statistics
        return get_params(self.model), len(train_dataloader), self.trainer.callback_metrics

    def evaluate(self, parameters, config):
        ## Set the directories
        #print(f"Config: {config}\t{self.cfg['callbacks']['model_checkpoint']['dirpath']}")
        #self.set_server_round_dir(config["server_round"])

        self.callbacks: List[Callback] = utils.instantiate_callbacks(self.cfg.callbacks)

        self.logger: List[LightningLoggerBase] = utils.instantiate_loggers(self.cfg.get("logger"))

        ## Create Trainer Object
        self.trainer: Trainer = hydra.utils.instantiate(self.cfg.trainer, callbacks=self.callbacks, logger=self.logger)

        set_params(self.model, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])#int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        
        val_dataloader = self.datamodule.val_dataloader(num_workers=num_workers)
        
        _ = self.trainer.validate(model=self.model, dataloaders=val_dataloader)

        return self.trainer.callback_metrics["val/loss"].item(), len(val_dataloader), self.trainer.callback_metrics
    

def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
    cfg: DictConfig,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        #print(f"Server Round: {server_round}")
        ## Instantiate model
        model = hydra.utils.instantiate(cfg.model)
        
        set_params(model, parameters)

        ## Create Data Module and call setup
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.setup("test")

        logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

        ## Trainer Test Function
        trainer.test(model=model, datamodule=datamodule)

        ## Save Global Model
        
        model_dir = cfg.paths.output_dir + '/global_model_checkpoints/'
        os.makedirs(model_dir, exist_ok=True)
        test_auc_roc = str(round(trainer.callback_metrics['test/aucroc'].item(), 4)*100).replace('.', '_')
        model_path = model_dir + f"round__{server_round}__{test_auc_roc}.pt"
        torch.save({
                    'epoch': server_round,
                    'model_state_dict': model.state_dict(),
                    'test_loss': trainer.callback_metrics['test/loss'],
                    'test_aucroc': trainer.callback_metrics['test/aucroc']
                    }, model_path)
        
        # return statistics
        return trainer.callback_metrics['test/loss'], trainer.callback_metrics

    return evaluate