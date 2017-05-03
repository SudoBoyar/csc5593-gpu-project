# csc5593-gpu-project

Team 7 - Investigation of the GPU Memory Hierarchy.

The applications can be compiled by running `make heracles`, `make hydra`, or `make dozer`, for whatever system you are on.

Three parameters need to be provided to run an application:

- -d [n] - The number of dimensions
- -i [n] - The number of iterations
- -n [n] - The size of the matrix (in all dimensions).

The applications are:

1. Naive - naive_gpu
1. Blocked - blocked_gpu
1. Overlapped Tiling - overlapped_gpu
1. Cached Plane Sweep - cps_gpu
1. Cached Plane Sweep Shared Memory - cpssm_gpu

The Naive, Blocked, and Overlapped Tiling algorithms can be run for 1, 2, or 3 dimensions.  
Both versions of the Cached Plane Sweep algorithm can only be run in 3 dimensions.  
All of our data was gathered using only the 3 dimensional versions.

There are run scripts provided to run all of the benchmarks on heracles and hydra. 
They are self contained and will simply run their full suite of tests.

#### Custom Configurations

There are additional values defined in the files that affect their behavior.

TILE_WIDTH is the total computation size in the X dimension for each block  
TILE_HEIGHT is the total computation size in the Y dimension for each block  
TILE_DEPTH is the total computation size in the Z dimension for each block  

PER_THREAD_X is the number of computations in the X dimension each thread performs  
PER_THREAD_Y is the number of computations in the Y dimension each thread performs  
PER_THREAD_Z is the number of computations in the Z dimension each thread performs  

The number of threads that is allocated is determined by dividing the first by the second of these two variables.  

For example:  
TILE_WIDTH = 32; TILE_HEIGHT = 4; TILE_DEPTH = 2;   
PER_THREAD_X = 4; PER_THREAD_Y = 1; PER_THREAD_Z = 2;

Will produce the the thread allocation:  
`dim3 threads(8, 4, 1);`

And a 32x32x32 matrix would have the grid dimensions:  
`dim3 blocks(1, 8, 16)`

The applications must be compiled after altering these values. 
Only the Blocked and Overlapped Tiling use all of these values. 
Those that are available to define are in a block near the top of their respective files.