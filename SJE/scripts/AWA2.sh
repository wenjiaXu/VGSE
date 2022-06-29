#!/bin/bash
device="0"
DATASET=AWA2

echo run VGSE-SMO ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding VGSE_SMO \
--nepoch 200 --classifier_lr 0.00001 --dataset ${DATASET}  --batch_size 64 --manualSeed 5214

echo run VGSE-SMO GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding VGSE_SMO \
--nepoch 200 --classifier_lr 0.0002 --dataset ${DATASET}  --batch_size 64 --gzsl --nclass_all 50 --nclass_all 50 --calibrated_stacking 0.95 --manualSeed 5214

echo run w2v ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding  word2vec \
--nepoch 200 --classifier_lr 0.00001 --dataset ${DATASET}  --batch_size 64 --manualSeed 5214

echo run w2v GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding word2vec \
--nepoch 200 --classifier_lr 0.0002 --dataset ${DATASET}  --batch_size 64 --gzsl --nclass_all 50 --nclass_all 50 --calibrated_stacking 0.95 --manualSeed 5214
