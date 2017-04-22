
#include <iostream>
#include <unistd.h>

// Shorthand for formatting usage options
#define fpe(msg) fprintf(stderr, "\t%s\n", msg);

#define HANDLE_ERROR(err)  ( HandleError( err, __FILE__, __LINE__ ) )

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

struct Args {
    bool debug = false;
    bool sequential = false;
    bool blocked = false;
    bool overlapped = false;
    // Data attributes
    int size = 1024, dimensions = 2, alloc_size;
    int xSize = 1, ySize = 1, zSize = 1;
    int xBlockSize = 1, yBlockSize = 1, zBlockSize = 1, tBlockSize;
    // Run attributes
    int grid_size = 1, block_count = -1, thread_count = -1, iterations = 1000;
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
    float* elements;
} Matrix;

Matrix initialize_matrix(int dimensions, int width, int height = 1, int depth = 1) {
    Matrix data;

    if (dimension == 1 && width > 1) {
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
        // Bad data
    }

    return data;
}

__global__ void jacobi1d_naive(Matrix data, Matrix result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float newValue;

    if (id > 0 && id < size - 1) {
        newValue = (data.elements[id - 1] + data.elements[id] + data.elements[id + 1]) / 3;
        __syncthreads();
        result.elements[id] = newValue;
    } else {
        // Edge or outside completely, do not change.
        __syncthreads();
    }
}

__global__ void jacobi2d_naive(Matrix data, Matrix result) {
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * blockDim.x + threadCol;
    int y = blockRow * blockDim.y + threadRow;

    int index = x + y * data.width;
    int xPrev = (x - 1) + y * data.width;
    int xNext = (x + 1) + y * data.width;
    int yPrev = x + (y - 1) * data.width;
    int yNext = x + (y + 1) * data.width;

    float newValue;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1) {
        newValue =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext]
            ) * 0.2;
        __syncthreads();
        result.elements[index] = newValue;
    } else {
        // Edge or beyond, do not change.
        __syncthreads();
    }
}

__global__ void jacobi3d_naive(Matrix data, Matrix result) {
    int threadDep = threadIdx.z;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockDep = blockIdx.z;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int x = blockCol * TILE_WIDTH + threadCol;
    int y = blockRow * TILE_HEIGHT + threadRow;
    int z = blockDep * TILE_DEPTH + threadDep;

    int xySurface = data.width * data.height;
    int zTemp = z * xySurface;
    int yTemp = y * data.width;

    int index = x + yTemp + zTemp; // x + y * data.width + z * data.width * data.height;
    int xPrev = (x - 1) + yTemp + zTemp; // (x-1) + y * data.width + z * data.width * data.height;
    int xNext = (x + 1) + yTemp + zTemp; // (x+1) + y * data.width + z * data.width * data.height;
    int yPrev = x + yTemp - data.width + zTemp; // x + (y-1) * data.width + z * data.width * data.height;
    int yNext = x + yTemp + data.width + zTemp; // x + (y+1) * data.width + z * data.width * data.height;
    int zPrev = x + yTemp + zTemp - xySurface; // x + y * data.width + (z-1) * data.width * data.height;
    int zNext = x + yTemp + zTemp + xySurface; // x + y * data.width + (z+1) * data.width * data.height;

    float newValue;

    if (x > 0 && x < data.width - 1 && y > 0 && y < data.height - 1 && z > 0 && z < data.depth - 1) {
        newValue =
            (
                data.elements[index] +
                data.elements[xPrev] +
                data.elements[xNext] +
                data.elements[yPrev] +
                data.elements[yNext] +
                data.elements[zPrev] +
                data.elements[zNext]
            ) / 7;
        __syncthreads();
        result.elements[index] = newValue;
    } else {
        // Edge or beyond, do not change.
        __syncthreads();
    }
}

void jacobi_naive(Args args, Matrix A, Matrix B) {
    if (args.dimensions == 1) {
        //dim3 blocks(args.grid_size);
        //dim3 threads(args.size/args.grid_size);
        dim3 blocks(args.size/32);
        dim3 threads(min(args.size, 32));

        for (int t = 0; t < args.iterations; t++) {
            jacobi1d_naive<<<blocks, threads>>>(A, B);
        }
    } else if (args.dimensions == 2) {
        dim3 blocks(args.size/16, args.size/16);
        dim3 threads(16, 16);
        for (int t = 0; t < args.iterations; t++) {
            jacobi2d_naive<<<blocks, threads>>>(A, B);
        }
    } else {
        dim3 blocks(args.size/8, args.size/8, args.size/8);
        dim3 threads(8, 8, 8);
        for (int t = 0; t < args.iterations; t++) {
            jacobi3d_naive<<<blocks, threads>>>(A, B);
        }
    }
}

int main(int argc, char *argv[]) {
    Args args = parse_arguments(argc, argv);
    Matrix A, B;
    A = initialize_matrix(args.dimensions, x, y, z);
    B = initialize_matrix(args.dimensions, x, y, z);

    atexit(cleanupCuda);

    //if (args.debug) { print_data(data, args.size, args.dimensions); }
    jacobi_naive(args, A, B);
    if (args.debug) { print_data(data, args.size, args.dimensions); }
}