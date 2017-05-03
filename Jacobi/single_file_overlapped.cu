#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>

using namespace std;

// Shorthand for formatting and printing usage options to stderr
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

// Shorthand for handling CUDA errors.
#define HANDLE_ERROR(err)  ( HandleError( err, __FILE__, __LINE__ ) )

/**
 * DEFINED VALUES HERE
 */

#define TILE_WIDTH 32
#define TILE_HEIGHT 1
#define TILE_DEPTH 1
#define TILE_AGE 4
#define PER_THREAD_X 1
#define PER_THREAD_Y 1
#define PER_THREAD_Z 1


/*****************
 * CUDA Utilites *
 *****************/

void HandleError(cudaError_t err, const char *file, int line) {
    //
    // Handle and report on CUDA errors.
    //
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);

        exit(EXIT_FAILURE);
    }
}

void checkCUDAError(const char *msg, bool exitOnError) {
    //
    // Check cuda error and print result if appropriate.
    //
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        if (exitOnError) {
            exit(-1);
        }
    }
}

void cleanupCuda(void) {
    //
    // Clean up CUDA resources.
    //

    //
    // Explicitly cleans up all runtime-related resources associated with the
    // calling host thread.
    //
    HANDLE_ERROR(
        cudaThreadExit()
    );
}

/*********************
 * End CUDA Utilites *
 *********************/

struct Args {
    bool debug;
    bool sequential;
    bool blocked;
    bool overlapped;
    // Data attributes
    int size, dimensions, alloc_size;
    int xSize, ySize, zSize;
    int xBlockSize, yBlockSize, zBlockSize, tBlockSize;
    // Run attributes
    int grid_size, block_count, thread_count, iterations;
};

void usage(char *prog_name, string msg) {
    if (msg.size() > 0) {
        fputs(msg.c_str(), stderr);
    }

    fprintf(stderr, "%s\n", prog_name);
    fprintf(stderr, "Options are:\n");
    fpe("-n<size> Set data size (default: 1024)");
    fpe("-d<dims> Set number of data dimensions (1, 2, or 3) (default: 2)");
    fpe("-g<size> Set grid size");
    fpe("-b<num>  Set block count");
    fpe("-t<num>  Set thread count");
    fpe("-i<iter> Number of iterations to perform (default: 1000)");
    fpe("-x<size> X Dimension");
    fpe("-y<size> Y Dimension");
    fpe("-z<size> Z Dimension");
    fpe("-T<size> T Dimension");
    fpe("-S       Execute sequential, CPU version");
    fpe("-B       Execute blocked sequential, CPU version");
    fpe("-O       Execute sequential overlapped tiling, CPU version");
    fpe("-D       Print debug info");
    fpe("-h       Print usage info (this message)");
    exit(EXIT_FAILURE);
}

Args parse_arguments(int argc, char *argv[]) {
    Args args = Args();
    args.debug = false;
    args.sequential = false;
    args.blocked = false;
    args.overlapped = false;
    args.size = 1024;
    args.dimensions = 2;
    args.xSize = args.ySize = args.zSize = 1;
    args.xBlockSize = args.yBlockSize = args.zBlockSize = 1;
    args.grid_size = 1;
    args.block_count = -1;
    args.thread_count = -1;
    args.iterations = 1000;

    int opt;
    // Parse args
    while ((opt = getopt(argc, argv, "n:d:g:b:t:i:x:y:z:T:hSBOD")) != -1) {
        switch (opt) {
            case 'D':
                args.debug = true;
                break;
            case 'S':
                args.sequential = true;
                break;
            case 'B':
                args.blocked = true;
                break;
            case 'O':
                args.overlapped = true;
                break;
            case 'n':
                args.size = atoi(optarg);
                break;
            case 'd':
                args.dimensions = atoi(optarg);
                break;
            case 'g':
                args.grid_size = atoi(optarg);
                break;
            case 'b':
                args.block_count = atoi(optarg);
                break;
            case 't':
                args.thread_count = atoi(optarg);
                break;
            case 'i':
                args.iterations = atoi(optarg);
                break;
            case 'x':
                args.xBlockSize = atoi(optarg);
                break;
            case 'X':
                args.xSize = atoi(optarg);
                break;
            case 'y':
                args.yBlockSize = atoi(optarg);
                break;
            case 'Y':
                args.ySize = atoi(optarg);
                break;
            case 'z':
                args.zBlockSize = atoi(optarg);
                break;
            case 'Z':
                args.zSize = atoi(optarg);
                break;
            case 'T':
                args.tBlockSize = atoi(optarg);
                break;
            case 'h':
                usage(argv[0], "");
                break;
            default:
                usage(argv[0], "Unrecognized option\n");
        }
    }

    // check sizes
    if (args.size <= 0) {
        cout << "Data size must be larger than 0" << endl;
        exit(EXIT_FAILURE);
    }

    if (args.dimensions <= 0 || args.dimensions >= 4) {
        cerr << "Data must be 1, 2, or 3 dimensions" << endl;
        exit(EXIT_FAILURE);
    }

    // Calculations
    if (args.dimensions == 1) {
        args.alloc_size = args.size;
    } else if (args.dimensions == 2) {
        args.alloc_size = args.size * args.size;
    } else {
        args.alloc_size = args.size * args.size * args.size;
    }

    if (args.thread_count > 0) {
        args.block_count = args.alloc_size / args.thread_count;
    } else if (args.block_count > 0) {
        args.thread_count = args.alloc_size / args.block_count;
    } else {
        args.thread_count = 16;
        args.block_count = args.alloc_size / args.thread_count;
    }

    return args;
}

typedef struct {
    int dimensions;
    int height;
    int width;
    int depth;
    float *elements;
} Matrix;

Matrix initialize_matrix(int dimensions, int width, int height = 1, int depth = 1) {
    Matrix data;

    if (dimensions == 1 && width > 1) {
        data.width = width;
        data.height = 1;
        data.depth = 1;
        data.elements = (float *) malloc(width * sizeof(float));

        data.elements[0] = 1.0;
        data.elements[width - 1] = 1.0;
    } else if (dimensions == 2 && width > 1 && height > 1) {
        data.width = width;
        data.height = height;
        data.depth = 1;
        data.elements = (float *) malloc(width * height * sizeof(float));

        for (int y = 0; y < height; y += height - 1) {
            for (int x = 0; x < width; x++) {
                data.elements[y * width + x] = 1.0;
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x += width - 1) {
                data.elements[y * width + x] = 1.0;
            }
        }
    } else if (dimensions == 3 && width > 1 && height > 1 && depth > 1) {
        data.width = width;
        data.height = height;
        data.depth = depth;
        data.elements = (float *) malloc(width * height * depth * sizeof(float));

        for (int z = 0; z < depth; z++) {
            // X = 0 & N planes
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x += width - 1) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }

            // Y = 0 & N planes
            for (int y = 0; y < height; y += height - 1) {
                for (int x = 0; x < width; x++) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }
        }

        // Z = 0 & N planes
        for (int z = 0; z < depth; z += depth - 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data.elements[z * width * height + y * width + x] = 1.0;
                }
            }
        }
    } else {
        fprintf(stderr, "Improper dimension or size.");
        exit(1);
    }

    return data;
}

/****************
 * CUDA KERNELS *
 ****************/

#define BLOCK_DIM_X TILE_WIDTH/PER_THREAD_X
#define BLOCK_DIM_Y TILE_HEIGHT/PER_THREAD_Y
#define BLOCK_DIM_Z TILE_DEPTH/PER_THREAD_Z

// ceil integer division, have to use the BLOCK_DIM_ definitions rather than the defines themselves or it won't work
#define PER_THREAD_OVERLAPPED_COUNT_X (TILE_AGE + TILE_WIDTH/PER_THREAD_X - 1) / (TILE_WIDTH/PER_THREAD_X)
#define PER_THREAD_OVERLAPPED_COUNT_Y (TILE_AGE + TILE_HEIGHT/PER_THREAD_Y - 1) / (TILE_HEIGHT/PER_THREAD_Y)
#define PER_THREAD_OVERLAPPED_COUNT_Z (TILE_AGE + TILE_DEPTH/PER_THREAD_Z - 1) / (TILE_DEPTH/PER_THREAD_Z)

#define PER_THREAD_COMBINED_ITERATIONS_X (PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X)
#define PER_THREAD_COMBINED_ITERATIONS_Y (PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y + PER_THREAD_OVERLAPPED_COUNT_Y)
#define PER_THREAD_COMBINED_ITERATIONS_Z (PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z + PER_THREAD_OVERLAPPED_COUNT_Z)

__global__ void jacobi1d(Matrix data, Matrix result) {
    int threadCol = threadIdx.x;
    int blockCol = blockIdx.x;

    int globalX[PER_THREAD_COMBINED_ITERATIONS_X];
    int sharedX[PER_THREAD_COMBINED_ITERATIONS_X];

    // Shared and local data arrays
    __shared__ float shared[2][(TILE_AGE + TILE_WIDTH + TILE_AGE)];
    int sharedXMax = TILE_AGE + TILE_WIDTH + TILE_AGE - 1;
    int tCurr = 0;
    int tPrev = 1;

    // Some useful bits of info
    int globalBlockStart = blockCol * TILE_WIDTH;
    // Use >= comparison
    int globalBlockReadStart = max(0, globalBlockStart - TILE_AGE);
    // Use <= comparison
    int globalBlockReadEnd = min(data.width - 1, globalBlockStart + TILE_WIDTH + TILE_AGE);

    // Indexes in overlapped region left of the block
#pragma unroll
    for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + threadCol - (PER_THREAD_OVERLAPPED_COUNT_X - x) * BLOCK_DIM_X;
        int globX = globalBlockStart + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
        // Locations inside the block
        int sharX = TILE_AGE + threadCol + BLOCK_DIM_X * (x - PER_THREAD_OVERLAPPED_COUNT_X);
        int globX = globalBlockStart + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + TILE_WIDTH + threadCol + BLOCK_DIM_X * (x - (PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X));
        int globX = globalBlockStart + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

    __syncthreads();

    /**
     * Global Memory:
     *
     *   Block 0   Block 1   Block 2   Block 3   Block 4
     * | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ |
     *
     * If we're block 2, we need:
     *
     *   Block 0   Block 1   Block 2   Block 3   Block 4
     * | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ |
     *                     |  this   |
     *
     * And for a tile age of AGE we also need:
     *
     *   Block 0   Block 1   Block 2   Block 3   Block 4
     * | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ |
     *              | this |         | this |
     *
     * So what we end up with is
     *
     *   Block 0   Block 1   Block 2   Block 3   Block 4
     * | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ | _ _ _ _ |
     *              | AGE  |  TLSIZE | AGE  |
     *
     * TILE_AGE + TILE_SIZE + TILE_AGE
     */

    // Read the block data itself into shared memory, this will always coalesce nicely
#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
        shared[0][sharedX[x]] = data.elements[globalX[x]];
    }

    // Read the left overlapped data into shared memory
#pragma unroll
    for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        // Left hand side data
        int globX = globalX[x];
        if (globX >= globalBlockReadStart && globX <= globalBlockReadEnd) {
            shared[0][sharedX[x]] = data.elements[globX];
        }
    }

    // Read the right overlapped data into shared memory
#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        // Left hand side data
        int globX = globalX[x];
        if (globX >= globalBlockReadStart && globX <= globalBlockReadEnd) {
            shared[0][sharedX[x]] = data.elements[globX];
        }
    }

    /*
     * Calculate Values
     */
#pragma unroll
    for (int t = 1; t <= TILE_AGE; t++) {
        int tmp = tCurr;
        tCurr = tPrev;
        tPrev = tmp;
        __syncthreads();

        int iterationCalculateStart = max(globalBlockStart - TILE_AGE + t - 1, 0);
        int iterationCalculateEnd = min(globalBlockStart + TILE_WIDTH + TILE_AGE - t, data.width - 1);

        // First let's do the block itself, since that always plays nicely
#pragma unroll
        for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
            int globX = globalX[x];
            int sharX = sharedX[x];

            if (globX > iterationCalculateStart && globX < iterationCalculateEnd) {
                shared[tCurr][sharX] = (shared[tPrev][sharX] + shared[tPrev][sharX - 1] + shared[tPrev][sharX + 1]) / 3;
            } else if (sharX >= 0){
                shared[tCurr][sharX] = shared[tPrev][sharX];
            }
        }

        // Now the left overlapped regions
#pragma unroll
        for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
            int globX = globalX[x];
            int sharX = sharedX[x];

            if (globX > iterationCalculateStart && globX < iterationCalculateEnd) {
                shared[tCurr][sharX] = (shared[tPrev][sharX - 1] + shared[tPrev][sharX] + shared[tPrev][sharX + 1]) / 3;
            } else if (sharX >= 0){
                shared[tCurr][sharX] = shared[tPrev][sharX];
            }
        }

        // And the right overlapped regions
#pragma unroll
        for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
            int globX = globalX[x];
            int sharX = sharedX[x];

            if (globX > iterationCalculateStart && globX < iterationCalculateEnd) {
                shared[tCurr][sharX] = (shared[tPrev][sharX - 1] + shared[tPrev][sharX] + shared[tPrev][sharX + 1]) / 3;
            } else if (sharX >= 0){
                shared[tCurr][sharX] = shared[tPrev][sharX];
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
        result.elements[globalX[x]] = shared[tCurr][sharedX[x]];
    }
}

__global__ void jacobi2d(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Indexes so we don't have to recompute them.
    int globalIndex[PER_THREAD_COMBINED_ITERATIONS_Y][PER_THREAD_COMBINED_ITERATIONS_X];
    int globalX[PER_THREAD_COMBINED_ITERATIONS_X];
    int globalY[PER_THREAD_COMBINED_ITERATIONS_Y];
    int sharedX[PER_THREAD_COMBINED_ITERATIONS_X];
    int sharedY[PER_THREAD_COMBINED_ITERATIONS_Y];

    // Shared and local data arrays
    __shared__ float shared[2][TILE_AGE + TILE_HEIGHT + TILE_AGE][TILE_AGE + TILE_WIDTH + TILE_AGE];
    int sharedXMax = TILE_AGE + TILE_WIDTH + TILE_AGE - 1;
    int sharedYMax = TILE_AGE + TILE_HEIGHT + TILE_AGE - 1;
    int tCurr = 0;
    int tPrev = 1;

    // Some useful bits of info
    int globalBlockStartX = blockCol * TILE_WIDTH;
    int globalBlockStartY = blockRow * TILE_HEIGHT;
    // Use >= comparison
    int globalBlockReadStartX = max(0, globalBlockStartX - TILE_AGE);
    int globalBlockReadStartY = max(0, globalBlockStartY - TILE_AGE);
    // Use <= comparison
    int globalBlockReadEndX = min(data.width - 1, globalBlockStartX + TILE_WIDTH + TILE_AGE);
    int globalBlockReadEndY = min(data.height - 1, globalBlockStartY + TILE_HEIGHT + TILE_AGE);

    /*
     * Calculate indexes into the global and shared arrays
     */

    // X Indexes

    // Overlapped region to the left of the block
#pragma unroll
    for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + threadCol - (PER_THREAD_OVERLAPPED_COUNT_X - x) * BLOCK_DIM_X;
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
        // Locations inside the block
        int sharX = TILE_AGE + threadCol + BLOCK_DIM_X * (x - PER_THREAD_OVERLAPPED_COUNT_X);
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + TILE_WIDTH + threadCol + BLOCK_DIM_X * (x - (PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X));
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

    // Y Indexes

    // Overlapped region below block
#pragma unroll
    for (int y = 0; y < PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
        // Offset by TILE_AGE to make sure it's within the range since we're going back by TILE_AGE
        int sharY = TILE_AGE + threadRow - (PER_THREAD_OVERLAPPED_COUNT_Y - y) * BLOCK_DIM_Y;
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Main block
#pragma unroll
    for (int y = PER_THREAD_OVERLAPPED_COUNT_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y++) {
        int sharY = TILE_AGE + threadRow + BLOCK_DIM_Y * (y - PER_THREAD_OVERLAPPED_COUNT_Y);
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Above block
#pragma unroll
    for (int y = PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y + PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
        int sharY = TILE_AGE + TILE_HEIGHT + threadRow + BLOCK_DIM_Y * (y - (PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y));
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Global absolute index
#pragma unroll
    for (int y = 0; y < PER_THREAD_COMBINED_ITERATIONS_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_COMBINED_ITERATIONS_X; x++) {
            globalIndex[y][x] = globalX[x] + globalY[y] * data.width;
        }
    }

    /*
     * Copy into shared memory
     */
    // TODO: Break into main block and overlapped regions blocks so the main block can at least be coalesced
#pragma unroll
    for (int y = 0; y < PER_THREAD_COMBINED_ITERATIONS_Y; y++) {
#pragma unroll
        for (int x = 0; x < PER_THREAD_COMBINED_ITERATIONS_X; x++) {
            /*
             * We want to be doing block-contiguous reads, e.g. for 2x2 block dimension, 2 per thread for x and y
             * we want the read pattern to look like:
             *
             * 11|22
             * 11|22
             * -----
             * 33|44
             * 33|44
             *
             * Optimizing the width for reads is the responsibility of the calling code.
             */
            if (globalX[x] >= 0 && globalX[x] < data.width && globalY[y] >= 0 && globalY[y] < data.height) {
                shared[0][sharedY[y]][sharedX[x]] = data.elements[globalIndex[y][x]];
            }
        }
    }

    /*
     * Calculate Values
     */
    // TODO Brevity and clarity might be better than this mismatched thing after all
#pragma unroll
    for (int t = 1; t <= TILE_AGE; t++) {
        int tmp = tCurr;
        tCurr = tPrev;
        tPrev = tmp;

        __syncthreads();

        int calculateStartX = max(globalBlockStartX - TILE_AGE + t - 1, 0);
        int calculateEndX = min(globalBlockStartX + TILE_WIDTH + TILE_AGE - t, data.width - 1);
        int calculateStartY = max(globalBlockStartY - TILE_AGE + t - 1, 0);
        int calculateEndY = min(globalBlockStartY + TILE_HEIGHT + TILE_AGE - t, data.height - 1);

#pragma unroll
        for (int y = PER_THREAD_OVERLAPPED_COUNT_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y++) {
            int globY = globalY[y];
            int sharY = sharedY[y];

            // First the main block since that's nicely laid out
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX &&
                    globY > calculateStartY && globY < calculateEndY) {
                    // Calculate new value
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // Now the left overlapped regions
#pragma unroll
            for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // And the right overlapped regions
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }
        }

        // Now the overlapped region below the block
#pragma unroll
        for (int y = 0; y < PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
            int globY = globalY[y];
            int sharY = sharedY[y];
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    // Calculate new value
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // Now the left and below overlapped region
#pragma unroll
            for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // And the right and below overlapped region
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }
        }

        // Overlapped region above the block
#pragma unroll
        for (int y = PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y + PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
            int globY = globalY[y];
            int sharY = sharedY[y];

#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    // Calculate new value
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // Now the left and below overlapped region
#pragma unroll
            for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }

            // And the right and below overlapped region
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
                int globX = globalX[x];
                int sharX = sharedX[x];

                if (globX > calculateStartX && globX < calculateEndX && globY > calculateStartY && globY < calculateEndY) {
                    shared[tCurr][sharY][sharX] =
                        (
                            shared[tPrev][sharY][sharX - 1] +
                            shared[tPrev][sharY][sharX] +
                            shared[tPrev][sharY][sharX + 1] +
                            shared[tPrev][sharY - 1][sharX] +
                            shared[tPrev][sharY + 1][sharX]
                        ) * 0.2f;
                } else if (sharX >= 0 && sharY >=0){
                    shared[tCurr][sharY][sharX] = shared[tPrev][sharY][sharX];
                }
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int y = PER_THREAD_OVERLAPPED_COUNT_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y++) {
        int sharY = sharedY[y];
        int globY = globalY[y];
#pragma unroll
        for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
            int sharX = sharedX[x];
            int globX = globalX[x];

            if (globX >= 0 && globX < data.width && globY >= 0 && globY < data.height) {
                result.elements[globalIndex[y][x]] = shared[tCurr][sharY][sharX];
            }
        }
    }
}

__global__ void jacobi3d(Matrix data, Matrix result) {
    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;
    int threadDep = threadIdx.z;
    int blockCol = blockIdx.x;
    int blockRow = blockIdx.y;
    int blockDep = blockIdx.z;

    // Indexes so we don't have to recompute them.
    int globalIndex[PER_THREAD_COMBINED_ITERATIONS_Z][PER_THREAD_COMBINED_ITERATIONS_Y][PER_THREAD_COMBINED_ITERATIONS_X];
    int globalX[PER_THREAD_COMBINED_ITERATIONS_X];
    int globalY[PER_THREAD_COMBINED_ITERATIONS_Y];
    int globalZ[PER_THREAD_COMBINED_ITERATIONS_Z];
    int sharedX[PER_THREAD_COMBINED_ITERATIONS_X];
    int sharedY[PER_THREAD_COMBINED_ITERATIONS_Y];
    int sharedZ[PER_THREAD_COMBINED_ITERATIONS_Z];

    // Shared and local data arrays
    __shared__ float shared[2][TILE_AGE + TILE_DEPTH + TILE_AGE][TILE_AGE + TILE_HEIGHT + TILE_AGE][TILE_AGE + TILE_WIDTH + TILE_AGE];
    int sharedXMax = TILE_AGE + TILE_WIDTH + TILE_AGE - 1;
    int sharedYMax = TILE_AGE + TILE_HEIGHT + TILE_AGE - 1;
    int sharedZMax = TILE_AGE + TILE_DEPTH + TILE_AGE - 1;
    int tCurr = 0;
    int tPrev = 1;

    // Some useful bits of info
    int globalBlockStartX = blockCol * TILE_WIDTH;
    int globalBlockStartY = blockRow * TILE_HEIGHT;
    int globalBlockStartZ = blockDep * TILE_DEPTH;
    // Use >= comparison
    int globalBlockReadStartX = max(0, globalBlockStartX - TILE_AGE);
    int globalBlockReadStartY = max(0, globalBlockStartY - TILE_AGE);
    int globalBlockReadStartZ = max(0, globalBlockStartZ - TILE_AGE);
    // Use <= comparison
    int globalBlockReadEndX = min(data.width - 1, globalBlockStartX + TILE_WIDTH + TILE_AGE);
    int globalBlockReadEndY = min(data.height - 1, globalBlockStartY + TILE_HEIGHT + TILE_AGE);
    int globalBlockReadEndZ = min(data.depth - 1, globalBlockStartZ + TILE_DEPTH + TILE_AGE);

    /*
     * Calculate indexes into the global and shared arrays
     */

    // Overlapped region to the left of the block
#pragma unroll
    for (int x = 0; x < PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + threadCol - (PER_THREAD_OVERLAPPED_COUNT_X - x) * BLOCK_DIM_X;
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
        // Locations inside the block
        int sharX = TILE_AGE + threadCol + BLOCK_DIM_X * (x - PER_THREAD_OVERLAPPED_COUNT_X);
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

#pragma unroll
    for (int x = PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X + PER_THREAD_OVERLAPPED_COUNT_X; x++) {
        int sharX = TILE_AGE + TILE_WIDTH + threadCol + BLOCK_DIM_X * (x - (PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X));
        int globX = globalBlockStartX + sharX - TILE_AGE;
        if (sharX < 0 || sharX > sharedXMax || globX < 0 || globX > data.width - 1) {
            sharedX[x] = -1;
            globalX[x] = -1;
        } else {
            sharedX[x] = sharX;
            globalX[x] = globX;
        }
    }

    // Y Indexes

    // Overlapped region below block
#pragma unroll
    for (int y = 0; y < PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
        // Offset by TILE_AGE to make sure it's within the range since we're going back by TILE_AGE
        int sharY = TILE_AGE + threadRow - (PER_THREAD_OVERLAPPED_COUNT_Y - y) * BLOCK_DIM_Y;
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Main block
#pragma unroll
    for (int y = PER_THREAD_OVERLAPPED_COUNT_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y++) {
        int sharY = TILE_AGE + threadRow + BLOCK_DIM_Y * (y - PER_THREAD_OVERLAPPED_COUNT_Y);
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Above block
#pragma unroll
    for (int y = PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y + PER_THREAD_OVERLAPPED_COUNT_Y; y++) {
        int sharY = TILE_AGE + TILE_HEIGHT + threadRow + BLOCK_DIM_Y * (y - (PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y));
        int globY = globalBlockStartY + sharY - TILE_AGE;
        if (sharY < 0 || sharY > sharedYMax || globY < 0 || globY > data.height - 1) {
            sharedY[y] = -1;
            globalY[y] = -1;
        } else {
            sharedY[y] = sharY;
            globalY[y] = globY;
        }
    }

    // Z Indexes

    // Overlapped region in front of block
#pragma unroll
    for (int z = 0; z < PER_THREAD_OVERLAPPED_COUNT_Z; z++) {
        // Offset by TILE_AGE to make sure it's within the range since we're going back by TILE_AGE
        int sharZ = TILE_AGE + threadDep - (PER_THREAD_OVERLAPPED_COUNT_Z - z) * BLOCK_DIM_Z;
        // Remove the offset for the global index
        int globZ = globalBlockStartZ + sharZ - TILE_AGE;
        if (sharZ < 0 || sharZ > sharedZMax || globZ < 0 || globZ > data.depth - 1) {
            sharedZ[z] = -1;
            globalZ[z] = -1;
        } else {
            sharedZ[z] = sharZ;
            globalZ[z] = globZ;
        }
    }

    // Main block
#pragma unroll
    for (int z = PER_THREAD_OVERLAPPED_COUNT_Z; z < PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z; z++) {
        int sharZ = TILE_AGE + threadDep + BLOCK_DIM_Z * (z - PER_THREAD_OVERLAPPED_COUNT_Z);
        int globZ = globalBlockStartZ + sharZ - TILE_AGE;
        if (sharZ < 0 || sharZ > sharedZMax || globZ < 0 || globZ > data.depth - 1) {
            sharedZ[z] = -1;
            globalZ[z] = -1;
        } else {
            sharedZ[z] = sharZ;
            globalZ[z] = globZ;
        }
    }

    // Overlapped region behind block
#pragma unroll
    for (int z = PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z; z < PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z + PER_THREAD_OVERLAPPED_COUNT_Z; z++) {
        int sharZ = TILE_AGE + TILE_DEPTH + threadDep + BLOCK_DIM_Z * (z - (PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z));
        int globZ = globalBlockStartZ + sharZ - TILE_AGE;
        if (sharZ < 0 || sharZ > sharedZMax || globZ < 0 || globZ > data.depth - 1) {
            sharedZ[z] = -1;
            globalZ[z] = -1;
        } else {
            sharedZ[z] = sharZ;
            globalZ[z] = globZ;
        }
    }

    // Global absolute index
#pragma unroll
    for (int z = 0; z < PER_THREAD_COMBINED_ITERATIONS_Z; z++) {
        int zTemp = globalZ[z] * data.width * data.height;
#pragma unroll
        for (int y = 0; y < PER_THREAD_COMBINED_ITERATIONS_Y; y++) {
            int yTemp = globalY[y] * data.width;
#pragma unroll
            for (int x = 0; x < PER_THREAD_COMBINED_ITERATIONS_X; x++) {
                globalIndex[z][y][x] = globalX[x] + yTemp + zTemp;
            }
        }
    }

    /*
     * Copy into shared memory
     */
#pragma unroll
    for (int z = 0; z < PER_THREAD_COMBINED_ITERATIONS_Z; z++) {
#pragma unroll
        for (int y = 0; y < PER_THREAD_COMBINED_ITERATIONS_Y; y++) {
#pragma unroll
            for (int x = 0; x < PER_THREAD_COMBINED_ITERATIONS_X; x++) {
                if (globalX[x] >= 0 && globalX[x] < data.width &&
                    globalY[y] >= 0 && globalY[y] < data.height &&
                    globalZ[z] >= 0 && globalZ[z] < data.depth) {
                    shared[0][sharedZ[z]][sharedY[y]][sharedX[x]] = data.elements[globalIndex[z][y][x]];
                }
            }
        }
    }

#pragma unroll
    for (int t = 1; t <= TILE_AGE; t++) {
        int tmp = tCurr;
        tCurr = tPrev;
        tPrev = tmp;

        __syncthreads();

        int calculateStartX = max(globalBlockStartX - TILE_AGE + t - 1, 0);
        int calculateEndX = min(globalBlockStartX + TILE_WIDTH + TILE_AGE - t, data.width - 1);
        int calculateStartY = max(globalBlockStartY - TILE_AGE + t - 1, 0);
        int calculateEndY = min(globalBlockStartY + TILE_HEIGHT + TILE_AGE - t, data.height - 1);
        int calculateStartZ = max(globalBlockStartZ - TILE_AGE + t - 1, 0);
        int calculateEndZ = min(globalBlockStartZ + TILE_DEPTH + TILE_AGE - t, data.depth - 1);

#pragma unroll
        for (int z = 0; z < PER_THREAD_COMBINED_ITERATIONS_Z; z++) {
            int globZ = globalZ[z];
            int sharZ = sharedZ[z];
#pragma unroll
            for (int y = 0; y < PER_THREAD_COMBINED_ITERATIONS_Y; y++) {
                int globY = globalY[y];
                int sharY = sharedY[y];
#pragma unroll
                for (int x = 0; x < PER_THREAD_COMBINED_ITERATIONS_X; x++) {
                    int globX = globalX[x];
                    int sharX = sharedX[x];

                    if (globX > calculateStartX && globX < calculateEndX &&
                        globY > calculateStartY && globY < calculateEndY &&
                        globZ > calculateStartZ && globZ < calculateEndZ) {

                        shared[tCurr][sharZ][sharY][sharX] =
                            (
                                shared[tPrev][sharZ][sharY][sharX] +
                                shared[tPrev][sharZ][sharY][sharX - 1] +
                                shared[tPrev][sharZ][sharY][sharX + 1] +
                                shared[tPrev][sharZ][sharY - 1][sharX] +
                                shared[tPrev][sharZ][sharY + 1][sharX] +
                                shared[tPrev][sharZ - 1][sharY][sharX] +
                                shared[tPrev][sharZ + 1][sharY][sharX]
                            ) / 7;
                    } else if (sharX >= 0 && sharY >= 0 && sharZ >= 0) {
                        shared[tCurr][sharZ][sharY][sharX] = shared[tPrev][sharZ][sharY][sharX];
                    }
                }
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int z = PER_THREAD_OVERLAPPED_COUNT_Z; z < PER_THREAD_OVERLAPPED_COUNT_Z + PER_THREAD_Z; z++) {
        int sharZ = sharedZ[z];
        int globZ = globalZ[z];
#pragma unroll
        for (int y = PER_THREAD_OVERLAPPED_COUNT_Y; y < PER_THREAD_OVERLAPPED_COUNT_Y + PER_THREAD_Y; y++) {
            int sharY = sharedY[y];
            int globY = globalY[y];
#pragma unroll
            for (int x = PER_THREAD_OVERLAPPED_COUNT_X; x < PER_THREAD_OVERLAPPED_COUNT_X + PER_THREAD_X; x++) {
                int sharX = sharedX[x];
                int globX = globalX[x];

                if (globX >= 0 && globY >= 0 && globZ >= 0) {
                    result.elements[globalIndex[z][y][x]] = shared[tCurr][sharZ][sharY][sharX];
                }
            }
        }
    }
}

/********************
 * END CUDA KERNELS *
 ********************/

Matrix initialize_device(Matrix A, bool copyToDevice) {
    Matrix deviceA;

    deviceA.width = A.width;
    deviceA.height = A.height;
    deviceA.depth = A.depth;
    deviceA.dimensions = A.dimensions;

    size_t sizeA = A.width * A.height * A.depth * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void **) &deviceA.elements, sizeA));
    if (copyToDevice) {
        HANDLE_ERROR(cudaMemcpy(deviceA.elements, A.elements, sizeA, cudaMemcpyHostToDevice));
    }

    return deviceA;
}

void callKernel(Args args, Matrix A, Matrix B) {
    Matrix deviceA, deviceB;

    deviceA = initialize_device(A, true);
    deviceB = initialize_device(B, false);

    if (args.dimensions == 1) {
        dim3 blocks(max(args.size / TILE_WIDTH, 1));
        dim3 threads(TILE_WIDTH / PER_THREAD_X);

        for (int t = 0; t < args.iterations / TILE_AGE; t++) {
            jacobi1d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi1d", true);
            swap(deviceA, deviceB);
        }
    } else if (args.dimensions == 2) {
        dim3 blocks(max(args.size / TILE_WIDTH, 1), max(args.size / TILE_HEIGHT, 1));
        dim3 threads(TILE_WIDTH / PER_THREAD_X, TILE_HEIGHT / PER_THREAD_Y);
        for (int t = 0; t < args.iterations / TILE_AGE; t++) {
            jacobi2d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi2d", true);
            swap(deviceA, deviceB);
        }
    } else {
        dim3 blocks(max(args.size / TILE_WIDTH, 1), max(args.size / TILE_HEIGHT, 1), max(args.size / TILE_DEPTH, 1));
        dim3 threads(TILE_WIDTH / PER_THREAD_X, TILE_HEIGHT / PER_THREAD_Y, TILE_DEPTH / PER_THREAD_Z);
        for (int t = 0; t < args.iterations / TILE_AGE; t++) {
            jacobi3d<<<blocks, threads>>>(deviceA, deviceB);
//            checkCUDAError("jacobi3d", true);
            swap(deviceA, deviceB);
        }
    }

    HANDLE_ERROR(cudaMemcpy(B.elements, deviceA.elements, A.width * A.height * A.depth * sizeof(float), cudaMemcpyDeviceToHost));
}

// Data output
void print_data(float *data, int size, int dimensions) {
    if (size > 32) {
        cerr << "Data too big to print\n" << endl;
        return;
    }

    if (dimensions == 1) {
        for (int x = 0; x < size; x++) {
            printf("%.3f ", data[x]);
        }
    } else if (dimensions == 2) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                printf("%.3f ", data[y * size + x]);
            }
            cout << endl;
        }
    } else if (dimensions == 3) {
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    printf("%.3f ", data[z * size * size + y * size + x]);
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    cout << endl << endl;
}

// Main
int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A, B;
    A = initialize_matrix(args.dimensions, args.size, args.size, args.size);
    B = initialize_matrix(args.dimensions, args.size, args.size, args.size);

    float runtime;
    struct timeval start, end;

    gettimeofday(&start, NULL);
    callKernel(args, A, B);
    gettimeofday(&end, NULL);
    runtime = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);
    printf("Processing Time: %4.4f milliseconds\n", runtime);
    if (args.debug) { print_data(B.elements, args.size, args.dimensions); }
}
