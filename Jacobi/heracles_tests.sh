#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 128 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 256 

ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 256

ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 256

ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 256

ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 256

ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 256
