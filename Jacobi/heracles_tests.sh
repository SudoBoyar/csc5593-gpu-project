#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Blocked Algorithm run for a data block of size 256x256x256"
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 128 
ssh node18 $DIR/blocked_gpu -d 3 -n 256 -i 256 

echo "Blocked Algorithm run for a data block of size 512x512x512"
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/blocked_gpu -d 3 -n 512 -i 256

echo "Naive Jacobi 3d Algorithm run for a data block of size 256x256x256"
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/naive_gpu -d 3 -n 256 -i 256

echo "Naive Jacobi 3d Algorithm run for a data block of size 512x512x512"
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/naive_gpu -d 3 -n 512 -i 256

echo "Cached Plane w/o Shared Memory Algorithm run for a data block of size 256x256x256"
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/cps_gpu -d 3 -n 256 -i 256

echo "Cached Plane w/o Shared Memory Algorithm run for a data block of size 512x512x512"
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/cps_gpu -d 3 -n 512 -i 256

echo "Cached Plane with Shared Memory Algorithm run for a data block of size 256x256x256"
ssh node18 $DIR/cpssm_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/cpssm_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/cpssm_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/cpssm_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/cpssm_gpu -d 3 -n 256 -i 256

echo "Cached Plane with Shared Memory Algorithm run for a data block of size 512x512x512"
ssh node18 $DIR/cpssm_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/cpssm_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/cpssm_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/cpssm_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/cpssm_gpu -d 3 -n 512 -i 256

echo "Overlapped Tiling Algorithm run for a data block of size 256x256x256"
ssh node18 $DIR/overlapped_gpu -d 3 -n 256 -i 16 
ssh node18 $DIR/overlapped_gpu -d 3 -n 256 -i 32 
ssh node18 $DIR/overlapped_gpu -d 3 -n 256 -i 64
ssh node18 $DIR/overlapped_gpu -d 3 -n 256 -i 128
ssh node18 $DIR/overlapped_gpu -d 3 -n 256 -i 256

echo "Overlapped Tiling Algorithm run for a data block of size 512x512x512"
ssh node18 $DIR/overlapped_gpu -d 3 -n 512 -i 16 
ssh node18 $DIR/overlapped_gpu -d 3 -n 512 -i 32 
ssh node18 $DIR/overlapped_gpu -d 3 -n 512 -i 64
ssh node18 $DIR/overlapped_gpu -d 3 -n 512 -i 128
ssh node18 $DIR/overlapped_gpu -d 3 -n 512 -i 256
