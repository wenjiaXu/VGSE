# VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning

The current project page provides [pytorch](http://pytorch.org/) code that implements the following [CVPR 2022](https://cvpr2022.thecvf.com/) paper:   
**Title:**      "VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning"    
**Authors:**     Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata    

**Abstract:**  
Human-annotated attributes serve as powerful semantic embeddings in zero-shot learning. However, their annotation process is labor-intensive and needs expert supervision. Current unsupervised semantic embeddings, i.e., word embeddings, enable knowledge transfer between classes. However, word embeddings do not always reflect visual similarities and result in inferior zero-shot performance.
We propose to discover semantic embeddings containing discriminative visual properties for zero-shot learning, without requiring any human annotation. Our model visually divides a set of images from seen classes into clusters of local image regions according to their visual similarity, and further imposes their class discrimination and semantic relatedness.
To associate these clusters with previously unseen classes, we use external knowledge, e.g., word embeddings and propose a novel class relation discovery module. Through quantitative and qualitative evaluation, we demonstrate that our model discovers semantic embeddings that model the visual properties of both seen and unseen classes. Furthermore, we demonstrate on three benchmarks that our visually-grounded semantic embeddings further improve performance over word embeddings across various ZSL models by a large margin.


## Requirements
Python 3.7.7

PyTorch = 1.8.1

## test

Please download the VGSE embeddings, w2v embeddings, and image features here: [data](https://drive.google.com/file/d/16PYq75orhr0UoE1OejjMfhullkr5ZZ9N/view?usp=sharing), and place it in *./SJE/*.
The data folder contains the files listed below (take AWA2 dataset as an example):

./data/AWA2/VGSE_SMO.mat:        Class embeddings learnt for AWA2 dataset.

./data/AWA2/word2vec_splits.mat: W2v embeddings learnt for AWA2 dataset.

./data/AWA2/res101.mat:          Image features.



To reproduce the results of applying our VGSE embeddings on SJE model, please run the following scripts:

sh ./SJE/scripts/AWA2.sh

sh ./SJE/scripts/CUB.sh

sh ./SJE/scripts/SUN.sh

<!-- - Data split and APN image features: please download the [data](https://drive.google.com/file/d/12ZsOxlkKU0IfXEfhB8NHRvHzfGFdwlhB/view?usp=sharing) folder and place it in *./data/*.

- Pre-trained models: please download the [pre-trained models](https://drive.google.com/file/d/1c5scuU0kZS5a9Rz3kf5T0UweCvOpGsh2/view?usp=sharing) and place it in *./pretrained_models/*.

## Code Structures
There are four parts in the code.
 - `model`: It contains the main files of the APN network.
 - `data`: The dataset split, as well as the APN feature extracted from our APN model.
 - `ABP`: The code from [ZSL_ABP](https://github.com/EthanZhu90/ZSL_ABP), we can reproduce the results of applying our APN feature on ABP model reported in the paper.
 - `pretrained_models`: The pretrained models.
 - `script`: The training scripts for APN, e.g., *./script/SUN_ZSL.sh*, etc. The training scripts for APN+ABP, i.e., *./script/SUN_APN_ABP.sh*, etc.

## Model zoo

We provide the trained ZSL model for three datasets as below:

 Dataset          | ZSL Accuracy   |  Download link | GZSL Accuracy |  Download link | 
 |  ----  | ----  | ----  | ----  | ----  |
| CUB          | 72.1                 |[Download](https://drive.google.com/file/d/1hPWNtbprwgrFlZmsauOV0mP0RCvekabA/view?usp=sharing) | 67.2 | [Download](https://drive.google.com/file/d/1mWxTwxWq1Nxt_c1XxA0isI7Tx4zVAH5A/view?usp=sharing)
| AWA2          | 68.6                 |[Download](https://drive.google.com/file/d/1ROau8p_si1qYhr5_gxdaIr_olen-DQp9/view?usp=sharing) | 67.4| [Download](https://drive.google.com/file/d/1_B4HyfQRyGw2KSZ_CIv8mBnEFlK5NRm7/view?usp=sharing)
| SUN          | 61.5                 |[Download](https://drive.google.com/file/d/1H-zB05WmfZytXDkdrptRLz-r--6Ta8dS/view?usp=sharing) |37.5| [Download](https://drive.google.com/file/d/1cRBv66A_YQUMqjexVOKF3_q3sgfLm62S/view?usp=sharing)

To perform evaluation, please download the model and place them into direction *./out/*， then run ./script/{dataset}_ZSL_eval.sh. 
 -->
##

If you feel this repo useful, please cite the following bib entry:

    @inproceedings{xu2022vgse,
      author    = {Xu, Wenjia and Xian, Yongqin and Wang, Jiuniu and Schiele, Bernt and Akata, Zeynep},
      title     = {VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year      = {2022}
    }

The code is under construction. If you have problems, feel free to reach me at xuwenjia16@mails.ucas.ac.cn

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
<!-- - [ZSL_ABP](https://github.com/EthanZhu90/ZSL_ABP)

- [CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning)

- [GEM-ZSL](https://github.com/osierboy/GEM-ZSL)

- [FEAT](https://github.com/Sha-Lab/FEAT)
 -->
