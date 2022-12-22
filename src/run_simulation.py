import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Dict, Optional, Callable

import hydra
from omegaconf import DictConfig
import flwr as fl
from flwr.common.typing import Scalar
from utils.data_split import process_traindata, partition_class, partition_volume, partition_feature
import pytorch_lightning as pl

from src import utils
from src.flwr_client import FlowerClient, get_evaluate_fn

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "server_round": server_round
    }
    return config

@utils.task_wrapper
def sim_run(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    client_resources = cfg.client_resources
    client_resources['num_gpus'] = 1 / cfg.pool_size

    client_config_file = cfg.client_config_name
    
    ## Number of dataset partions (= number of total clients)
    pool_size = cfg.pool_size 
    
    ## Data partitioning
    dataframe = process_traindata(cfg.partitions.train_path, cfg.partitions.policy)

    if cfg.partitions.key == 'volume':
        partitions = partition_volume(dataframe, cfg.partitions.mode, cfg.partitions.num_clients, cfg.partitions.scale, sample_percent=cfg.sample_percent)
    elif cfg.partitions.key == 'class':
        partitions = partition_class(dataframe, cfg.partitions.modeparams, cfg.partitions.mode, cfg.partitions.num_clients, cfg.partitions.exclusive, cfg.partitions.equal_num_samples, cfg.partitions.min_client_samples, cfg.sample_percent)
    elif cfg.partitions.key=="feature":
        partitions = partition_feature(dataframe, cfg.partitions.feature, cfg.partitions.num_partitions, mode=cfg.partitions.mode)     
    
    ## Configure the strategy
    strategy = hydra.utils.instantiate(cfg.strategy)
    strategy.evaluate_fn = get_evaluate_fn(cfg)
    strategy.on_fit_config_fn = fit_config

    hydra_runtime_log_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, client_config_file, partitions[int(cid)], hydra_runtime_log_dir, cfg['datamodule']['batch_size'])
    
    server_config = hydra.utils.instantiate(cfg.server_config)

    ## Start Simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=server_config,
        strategy=strategy,
        ray_init_args=cfg.ray_init_args,
    )

    return None, None

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="simulation")
def main(cfg: DictConfig) -> Optional[float]:
    
    _ = sim_run(cfg)

    return "YOLO"
    
if __name__ == "__main__":
    main()
    
