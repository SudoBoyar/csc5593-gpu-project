#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ssh node18 $DIR/blocked_gpu -d 3 -n 32 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 64 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 128 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 64
