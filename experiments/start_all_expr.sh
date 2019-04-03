#!/bin/bash
CUDA_DEVICE=$1
ID=$2
nohup python lt-iterative-cifar2.py test-all-dataparallel --batch_size 128 --epoch 300 --prune 15 --ft 50 --log 100 --device cuda:$CUDA_DEVICE > test-all-densenet.out 2> test-all-densenet.err &
echo $! > test-all-densenet.pid

