# ReLeaSE (Reinforcement Learning for Structural Evolution)
Deep Reinforcement Learning for de-novo Drug Design

### Currently works only under Linux

This is an official PyTorch implementation of Deep Reinforcement Learning for de-novo Drug Design aka ReLeaSE method.

## REQUIREMENTS:
In order to get started you will need:
* Modern NVIDIA GPU, [compute capability 3.5](https://developer.nvidia.com/cuda-gpus) of newer.
* [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
* [Pytorch 0.4.1](https://pytorch.org)
* [Tensorflow 1.8.0](https://www.tensorflow.org/install/) with GPU support
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [Scikit-learn](http://scikit-learn.org/)
* [Numpy](http://www.numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [Mordred](https://github.com/mordred-descriptor/mordred)

## Installation with Anaconda

If you installed your Python with Anacoda you can run the following commands to get started:
```bash
# Clone the reopsitory to your desired directory
git clone https://github.com/isayev/ReLeaSE.git
cd ReLeaSE
# Create new conda environment with Python 3.6
conda create -n release python=3.6
# Activate the environment
conda activate release
# Install conda dependencies
conda install --yes --file conda_requirements.txt
conda install -c rdkit rdkit nox cairo
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
# Instal pip dependencies
pip install pip_requirements.txt
# Add new kernel to the list of jupyter notebook kernels
python -m ipykernel install --user --name release --display-name ReLeaSE
```

## Demos

We uploaded several demos in a form of iPython notebooks:
* JAK2_min_max_demo.ipynb -- [JAK2](https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL2363062) pIC50 minimization and maximization
* LogP_optimization_demo.ipynb -- optimization of logP to be in a drug-like region 
from 0 to 5 according to [Lipinski's rule of five](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five).
* RecurrentQSAR-example-logp.ipynb -- training a Recurrent Neural Network to predict logP from SMILES
using [OpenChem](https://github.com/Mariewelt/OpenChem) toolkit.

**Disclaimer**: JAK2 demo uses Random Forest predictor instead of Recurrent Neural Network,
since the availability of the dataset with JAK2 activity data used in the
"Deep Reinforcement Learning for de novo Drug Design" paper is restricted under
the license agreement. So instead we use the JAK2 activity data downladed from
ChEMBL (CHEMBL2971) and curated. The size of this dataset is ~2000 data points,
which is not enough to build a reliable deep neural network. If you want to see
a demo with RNN, please checkout logP optimization.

## Citation
If you use this code or data, please cite:

### ReLeaSE method paper:
Mariya Popova, Olexandr Isayev, Alexander Tropsha. *Deep Reinforcement Learning for de-novo Drug Design*. Science Advances, 2018, Vol. 4, no. 7, eaap7885. DOI: [10.1126/sciadv.aap7885](http://dx.doi.org/10.1126/sciadv.aap7885)
