#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Blocked Algorithm run for a data block of size 256x256x255"
echo "# of iterations: 16"
echo "# of iterations: 32"
bpsh 12 $DIR/blocked_gpu -d 3 -n 256 -i 16 
bpsh 12 $DIR/blocked_gpu -d 3 -n 256 -i 32 
bpsh 12 $DIR/blocked_gpu -d 3 -n 256 -i 64
bpsh 12 $DIR/blocked_gpu -d 3 -n 256 -i 128 
bpsh 12 $DIR/blocked_gpu -d 3 -n 256 -i 256 

bpsh 12 $DIR/blocked_gpu -d 3 -n 512 -i 16 
bpsh 12 $DIR/blocked_gpu -d 3 -n 512 -i 32 
bpsh 12 $DIR/blocked_gpu -d 3 -n 512 -i 64
bpsh 12 $DIR/blocked_gpu -d 3 -n 512 -i 128
bpsh 12 $DIR/blocked_gpu -d 3 -n 512 -i 256

bpsh 12 $DIR/naive_gpu -d 3 -n 256 -i 16 
bpsh 12 $DIR/naive_gpu -d 3 -n 256 -i 32 
bpsh 12 $DIR/naive_gpu -d 3 -n 256 -i 64
bpsh 12 $DIR/naive_gpu -d 3 -n 256 -i 128
bpsh 12 $DIR/naive_gpu -d 3 -n 256 -i 256

bpsh 12 $DIR/naive_gpu -d 3 -n 512 -i 16 
bpsh 12 $DIR/naive_gpu -d 3 -n 512 -i 32 
bpsh 12 $DIR/naive_gpu -d 3 -n 512 -i 64
bpsh 12 $DIR/naive_gpu -d 3 -n 512 -i 128
bpsh 12 $DIR/naive_gpu -d 3 -n 512 -i 256

bpsh 12 $DIR/cps_gpu -d 3 -n 256 -i 16 
bpsh 12 $DIR/cps_gpu -d 3 -n 256 -i 32 
bpsh 12 $DIR/cps_gpu -d 3 -n 256 -i 64
bpsh 12 $DIR/cps_gpu -d 3 -n 256 -i 128
bpsh 12 $DIR/cps_gpu -d 3 -n 256 -i 256

bpsh 12 $DIR/cps_gpu -d 3 -n 512 -i 16 
bpsh 12 $DIR/cps_gpu -d 3 -n 512 -i 32 
bpsh 12 $DIR/cps_gpu -d 3 -n 512 -i 64
bpsh 12 $DIR/cps_gpu -d 3 -n 512 -i 128
bpsh 12 $DIR/cps_gpu -d 3 -n 512 -i 256

bpsh 12 $DIR/cpssm_gpu -d 3 -n 256 -i 16 
bpsh 12 $DIR/cpssm_gpu -d 3 -n 256 -i 32 
bpsh 12 $DIR/cpssm_gpu -d 3 -n 256 -i 64
bpsh 12 $DIR/cpssm_gpu -d 3 -n 256 -i 128
bpsh 12 $DIR/cpssm_gpu -d 3 -n 256 -i 256

bpsh 12 $DIR/cpssm_gpu -d 3 -n 512 -i 16 
bpsh 12 $DIR/cpssm_gpu -d 3 -n 512 -i 32 
bpsh 12 $DIR/cpssm_gpu -d 3 -n 512 -i 64
bpsh 12 $DIR/cpssm_gpu -d 3 -n 512 -i 128
bpsh 12 $DIR/cpssm_gpu -d 3 -n 512 -i 256

bpsh 12 $DIR/overlapped_gpu -d 3 -n 256 -i 16 
bpsh 12 $DIR/overlapped_gpu -d 3 -n 256 -i 32 
bpsh 12 $DIR/overlapped_gpu -d 3 -n 256 -i 64
bpsh 12 $DIR/overlapped_gpu -d 3 -n 256 -i 128
bpsh 12 $DIR/overlapped_gpu -d 3 -n 256 -i 256

bpsh 12 $DIR/overlapped_gpu -d 3 -n 512 -i 16 
bpsh 12 $DIR/overlapped_gpu -d 3 -n 512 -i 32 
bpsh 12 $DIR/overlapped_gpu -d 3 -n 512 -i 64
bpsh 12 $DIR/overlapped_gpu -d 3 -n 512 -i 128
bpsh 12 $DIR/overlapped_gpu -d 3 -n 512 -i 256
