#!/bin/bash
device="0"
DATASET=SUN

echo run VGSE_SMO on ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding VGSE_SMO \
--nepoch 100 --classifier_lr 0.001 --dataset ${DATASET}  --batch_size 64 --manualSeed 1143

echo run VGSE_SMO on GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding VGSE_SMO \
--nepoch 100 --classifier_lr 0.0005 --dataset ${DATASET}  --batch_size 64  --gzsl --calibrated_stacking 0.8 --manualSeed 1143

echo run w2v on ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding  word2vec \
--nepoch 100 --classifier_lr 0.001 --dataset ${DATASET}  --batch_size 64 --manualSeed 1143

echo run w2v on GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding word2vec \
--nepoch 100 --classifier_lr 0.0005 --dataset ${DATASET}  --batch_size 64  --gzsl --calibrated_stacking 0.8 --manualSeed 1143
