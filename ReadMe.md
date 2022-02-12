# Hypergraph Convolutional Network for Multi-behavior Recommendation

This is the PyTorch implementation for our KDD 2022 submission:

>Hypergraph Convolutional Network for Multi-behavior Recommendation.


## Introduction
HGMBR is a new multi-behavior recommendation framework based on hypergraph convolution.

## Installation
This repo is developed under Python 3.7.

To install the dependencies, run the following command:
```
pip install -r requirements.txt
```

## Example to Run the Codes
* Tmall dataset
```
python main.py GPU=[0]
```

Our code is based on [Hydra](https://github.com/facebookresearch/hydra).

For other advanced configurations and functions, such as multi-run, please refer to this [tutorial](https://hydra.cc/docs/intro/).

## Baselines

We release our implementation of several baseline models. To try them, run the following commands:

```
python main.py data=Tmall model=MBGCN lam=1e-5
```

## Dataset
We also provide the Tmall dataset for our paper. Kuaishou is coming soon.


## Pretrain Weights
We also upload the pretrained weights used in our experiments for your convenience.

If you prefer to get the weights by yourself, simply run the following commands:

```
python main.py data=Tmall model=MF
```
And you will find the weights in the 'outputs' folder.