#!/bin/bash
echo "-d " $1 " -n " $2 " -i " $3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ssh node18 $DIR/naive_gpu -d $1 -n $2 -i $3
ssh node18 $DIR/blocked_gpu -d $1 -n $2 -i $3
ssh node18 $DIR/overlapped_gpu -d $1 -n $2 -i $3
ssh node18 $DIR/cps_gpu -d $1 -n $2 -i $3
ssh node18 $DIR/cpssm_gpu -d $1 -n $2 -i $3
