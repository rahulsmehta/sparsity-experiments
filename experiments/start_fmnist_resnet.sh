#!/bin/bash
CUDA_DEVICE=$1
nohup python lt-iterative-fmnist-resnet.py test-fmnist-iterative-resnet-$2 --batch_size 128 --prune 15 --epoch 200 --ft 50 --log 100 --device cuda:$CUDA_DEVICE > test-fmnist-iterative-resnet-$2.out 2> test-fmnist-iterative-resnet-$2.err &
echo $! > test-fmnist-iterative-resnet-$2.pid