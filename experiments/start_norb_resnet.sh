#!/bin/bash
CUDA_DEVICE=$1
nohup python lt-iterative-norb-vgg.py test-norb-iterative-vgg --batch_size 64 --epoch 160 --ft 40 --log 100 --device cuda:$CUDA_DEVICE > test-norb-iterative-vgg.out 2> test-norb-iterative-vgg.err &
echo $! > test-norb-iterative-vgg.pid

