______________________________________________________________________

<div align="center">

#  ConVIRT in Federated Setup

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

This repository contains code to train the ConVIRT model on the MIMIC-CXR-JPG dataset and fine tune the pretrained image backbone for downstream image multi-label classification on the CheXpert dataset in centralized and federated learning setups.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/tjdevWorks/ConVIRT-Federated
cd ConVIRT-Federated

# [OPTIONAL] create conda environment
conda create -n convirt_fed python=3.7
conda activate convirt_fed

# install requirements
pip install -r requirements.txt
```

Pretraining the model with default configuration

```bash
python src/pretrain.py
```

Fine tuning model slurm configuration execution scripts are available [scripts/tejas/a100/](scripts/tejas/a100/)

An example of fine tuning in centralized setup:

```bash
# To use the ConVIRT pretrained model Image Backbone
python src/finetune_chexpert.py

# To use the ImageNet pretrained model image backbone
python src/finetune_chexpert.py --config-name=finetune_chexpert_imagenet
```

To execute the federated learning setups we have three data partitioning strategies in [configs/partitions/](configs/partitions/) volume, class, attribute.

An example of running a federated learning experiment:

```bash
# Runs a federated simulation on a single node with gpu using 4 clients for 100 rounds and paritioning logic for "class.yaml"
python src/run_simulation.py --config-name=prod_simulation server_config.num_rounds=100 pool_size=4 partitions=class partitions.num_clients=4 partitions.exclusive=False partitions.equal_num_samples=False task_name='fed_chexpert_class' job_name=fed_class_100_4_False_False datamodule.batch_size=256
```

You can override any parameter from command line like this

```bash
python src/finetune_chexpert.py trainer.max_epochs=20 datamodule.batch_size=64
```
## Citation
```bash
@article{DBLP:journals/corr/abs-2010-00747,
  author    = {Yuhao Zhang and
               Hang Jiang and
               Yasuhide Miura and
               Christopher D. Manning and
               Curtis P. Langlotz},
  title     = {Contrastive Learning of Medical Visual Representations from Paired
               Images and Text},
  journal   = {CoRR},
  volume    = {abs/2010.00747},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.00747},
  eprinttype = {arXiv},
  eprint    = {2010.00747},
  timestamp = {Fri, 20 Nov 2020 14:04:05 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-00747.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```