## Environment Setup Guide

We assume that you've followed [this repo](https://github.com/ziyueluocs/torch-tutorial) to setup your local programming environment. If not, here is a brief recap:

```bash
# 1. install miniconda
# 2. setup your conda environment
conda create -n reu_demo python=3.10 -y
conda activate reu_demo
# 3. install pytorch as instructed by https://pytorch.org/
#   the following command install torch for cuda 12.6 on my end
pip install torch torchvision
```

Additionally, install these dependencies:

```bash
# 4. install other dependencies
conda install scipy scikit-learn ipykernel
pip install matplotlib 
```