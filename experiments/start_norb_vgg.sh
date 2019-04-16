#!/bin/bash
CUDA_DEVICE=$1
nohup python lt-iterative-norb-vgg.py test-norb-iterative-vgg-$2 --batch_size 64 --prune 15 --epoch 160 --ft 40 --log 200 --device cuda:$CUDA_DEVICE > test-norb-iterative-vgg-$2.out 2> test-norb-iterative-vgg-$2.err &
echo $! > test-norb-iterative-vgg-$2.pid