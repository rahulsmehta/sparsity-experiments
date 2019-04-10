#!/bin/bash
CUDA_DEVICE=$1
nohup python lt-oneshot-norb.py test-norb-oneshot --batch_size 128 --epoch 200 --ft 50 --log 100 --device cuda:$CUDA_DEVICE >> test-norb-oneshot.out 2> test-norb-oneshot.err &
echo $! > test-norb-oneshot.pid

