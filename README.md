# Metrics Are Born At Sea But Stored in the Cloud

[![pytorch](https://img.shields.io/badge/PyTorch_1.12.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

*Repository accompanying my [NormConf 2022](https://normconf.com/) [talk]().*

## Introduction
In this repository you can retrace the design ideas from my talk with an end-to-end experiment, comparing [Graph Convolution](https://arxiv.org/abs/1609.02907) to [Graph Attention](https://arxiv.org/abs/1710.10903) on two small molecule classification datasets.

> If you notice slight differences between the code here and the screenshots in my talk - I refactored the `download.py` and `plot.ipynb` to make the repository self-contained and re-ran all experiments to test the repository.

For this demo, I used [Weights & Biases](https://wandb.ai/site) to log metrics but it should be straightforward to make the code work with other experiment trackers such as [Neptune](https://neptune.ai/product/experiment-tracking) or [Comet](https://www.comet.com/site/products/ml-experiment-tracking/).

To help you retrace the ideas from my talk, I added comments starting with "🤔" in the code and the `config.yaml`. Any questions, remarks or feedback can be logged as an issue on GitHub or via [e-mail](mailto:luismueller2309@gmail.com).

## Task
The example task we solve in this repository is a comparison of two popular models for graph classification, [Graph Convolution](https://arxiv.org/abs/1609.02907) and [Graph Attention](https://arxiv.org/abs/1710.10903). To enable a fair comparison, we train and evaluate the models on two small molecule datasets (*Proteins*, *Enzymes*, available [here](https://chrsmrrs.github.io/datasets/docs/datasets/)) and across multiple seeds.

The workflow follows the design ideas from my talk and comprises three steps:

1. Train and test both models on both datasets for multiple random seeds and log the metrics to a cloud service. (`main.py`)
2. Download all the metrics into one big `.csv` file for later processing. (`download.py`)
3. Analyze the experiment with different plots and a results table in LaTex. (`plot.ipynb`)

> Note that the actual task is unimportant to get my ideas across and just serves as an excuse to log actual metrics. However, if it gets you interested in graph learning, all the better 😇

## Installation
You can get the whole thing running in just three steps.

> I strongly recommend to set everything up with conda. Tested with Python 3.9.

1. [Install PyTorch](https://pytorch.org/get-started/locally/) (tested with `pytorch=1.12.1`).
2. [Install PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (matching your PyTorch version)[^1].
3. Install the remaining dependencies below with `pip`[^2].

```bash
pip install hydra-core torchmetrics wandb pandas seaborn ipykernel
```

[^1]: On Mac M1, it should be sufficient to run `pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html`. Make sure the wheel matches your torch version.

[^2]: I have also installed `black[jupyter]` to have [black formatted code](https://github.com/psf/black), but it is not stricly a dependency of the experiment code.

## Configuration
One of the dependencies of this repository is [Hydra](https://hydra.cc/), which turns both `main.py` (to train and test the models and log metrics) and `download.py` (to download the metrics from the cloud) into fully-configurable programs.

You can either adjust settings in the `config.yaml` file or via the command-line, e.g., to set a different seed, simply run
```bash
python main.py seed=42
```
A really powerful functionality of Hydra is the *multi-run*, which executes a script over all combinations of parameters. This is really useful here, because it lets us compute the experiment matrix
||Graph Convolution|Graph Attention|
|-|-|-|
|**Proteins**|x|x|
|**Enzymes**|x|x|

for multiple seeds with one command, namely:
```bash
python main.py -m hparams.Dataset=Enzymes,Proteins hparams.Model="Graph Convolution","Graph Attention" hparams.seed=91,17,44
```

> Make sure to set the config parameters `wandb.entity` and `wandb.project` to enable logging to a Weights & Biases instance.

In general, I recommend separating any local settings (such as wandb settings) from your experiments hyper-parameters, which is why the latter are specified under `hparams` in the `config.yaml`. As a result, we can now pass these params collectively to `wandb.init`:
```bash
wandb.init(
    ...
    config=dict(cfg.hparams),
    ...
)
```
which reduces the config-related code to a bare-minimum.