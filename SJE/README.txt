Evaluation code for the paper:
"VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning"

######################################################################################################################
Environment
Python 3.7.7
PyTorch = 1.8.1

######################################################################################################################
Example

To reproduce the results of applying our VGSE embeddings on SJE model, please run the following scripts:
sh scripts/AWA2.sh
sh scripts/CUB.sh
sh scripts/SUN.sh
######################################################################################################################
Illustration

This folder contains the files listed below.

data:
./data/AWA2/VGSE_SMO.mat:        Saving class embeddings learnt for AWA2 dataset.
./data/AWA2/word2vec_splits.mat: Saving w2v embeddings learnt for AWA2 dataset.
./data/AWA2/res101.mat:          Saving image features.

./data/CUB/VGSE_SMO.mat:         Saving class embeddings learnt for CUB dataset.
./data/CUB/word2vec_splits.mat: Saving w2v embeddings learnt for CUB dataset.
./data/CUB/res101.mat:           Saving image features.

./data/SUN/VGSE_SMO.mat:         Saving class embeddings learnt for SUN dataset.
./data/SUN/word2vec_splits.mat: Saving w2v embeddings learnt for SUN dataset.
./data/SUN/res101.mat:           Saving image features.

SJE model:
- SJE.py                Script for train SJE ZSL model

Other python files:
- classifier1.py        Network and loss
- classifier2.py        Network and loss
- util.py               Utility functions



