#!/bin/bash
nohup python data_parallel.py > data_parallel.out 2> data_parallel.err &
echo $! > data_parallel.pid