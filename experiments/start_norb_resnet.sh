#!/bin/bash
CUDA_DEVICE=$1
ID=$2
nohup python lt-iterative-norb-resnet.py test-norb-iterative-resnet-$ID --batch_size 128 --epoch 200 --ft 50 --log 100 --device cuda:$CUDA_DEVICE > test-norb-iterative-resnet-$ID.out 2> test-norb-iterative-resnet-$ID.err &
echo $! > test-norb-iterative-resnet-$ID.pid

