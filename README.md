# Neural Clot Simulation

Reimplementation of internal emulation of FEM (Abaqus) clot embolization simulations using equivariant neural fields.

## Installation

All code has been developed and tested with ``Python 3.12``. Below we list the minimal requirements for this repository. We also previde an easy-install script using [Astral's **uv**](https://docs.astral.sh/uv/).

### Requirements

This has the following requirements. 
```bash
torch==2.8.0+cu128
torch-cluster==1.6.3+pt28cu128
cython>=3.1.4
einops>=0.8.1
ml-collections>=1.1.0
omegaconf>=2.3.0
matplotlib>=3.10.7
tensorflow>=2.20.0 # for processing datasets
tqdm>=4.67.1
wandb>=0.22.2
```

### Install with UV

Simply run the following to download and install this repository with **uv**.
```bash
git clone https://github.com/thijskuipers1995/eq-ind.git
cd eq-ind
uv venv
# install torch and torch-cluster before calling uv sync
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv sync
```

## Usage
 We use `ml-collections` to manage configuration files. All settings present in the `*.yaml` config files can be overridden with with CLI. The data contracts are provided in `src/datasets/dataset_clot.py`.

### Minimal Example

To start a training session, run the following by providing the path to the correspondig configuration file:
```bash
uv run -m src.train --experiment=configs/experiments/eq-ind-flag-minimal.yaml
```
To perform a rollout on the first trajectory of the validation (default) set, provide the path to the output experiment folder:
```bash
uv run -m src.rollout experiment=outputs/eq-ind-flag-minimal/run trajectory=0
```

### Experiment Setup

Experiments are configured with the configuration files in the `configs/` directory, which has the following structure:
```
configs/
    ├── experiments
    |   ├── flag-minimal.yaml
    |   └── ...
    └── rollout.yaml # default rollout settings
```

#### Configuration files.
Configuration files define not only model/dataset/etc. settings, but also which model/dataset/etc. is loaded, specified with by the `_target_` field. This field simply contains the full import path to the desired module, e.g., `_target: src.models.MyModel` will import `MyModel` from `src.models`. Moreover, variable fields are supported for when settings are shared among multiple fields, e.g., we can set `scheduler.total_epochs: ${training.epochs}`.


