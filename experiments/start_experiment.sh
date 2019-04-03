#!/bin/bash

PRUNE_ITER=$1
CUDA='cuda:'$2
EXPR_NAME=lt-cifar-resnet18-$1

nohup python lt-iterative-cifar-resnet18.py $EXPR_NAME --epoch 200 --prune $PRUNE_ITER --ft 30 --log 100 --batch_size 128 --device $CUDA > $EXPR_NAME.out 2> $EXPR_NAME.err &

echo $! > $EXPR_NAME.pid