import os

# define DIMENSIONS %%DIMENSIONS%%
# define ITERATIONS %%ITERATIONS%%
# define SIZE %%SIZE%%
# define TILE_WIDTH %%TILE_WIDTH%%
# define TILE_HEIGHT %%TILE_HEIGHT%%
# define TILE_DEPTH %%TILE_DEPTH%%
# define PER_THREAD_X %%PER_THREAD_X%%
# define PER_THREAD_Y %%PER_THREAD_Y%%
# define PER_THREAD_Z %%PER_THREAD_Z%%

cases = {
    # SET 1
    # Data Dimensions: Variable
    # Block Dimensions: 32 x 32 x 1
    # Computations per core: 1
    'Set 1': {
        'DataDim-1024': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'DataDim-2048': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 2048,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'DataDim-4094': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 4096,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'DataDim-8192': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 8192,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'DataDim-16384': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
    },
    # SET 2
    # Data Dimensions: 1024 x 1024 x 1024
    # Block Dimensions: Variable
    # Computations per core: 1
    'Set2': {
        'Block-1x1024': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 1,
            'TILE_HEIGHT': 1024,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-2x512': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 2,
            'TILE_HEIGHT': 512,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-4x256': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 4,
            'TILE_HEIGHT': 256,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-8x128': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 8,
            'TILE_HEIGHT': 128,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-16x64': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 16,
            'TILE_HEIGHT': 64,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-32x32': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-64x16': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 64,
            'TILE_HEIGHT': 16,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-128x8': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 128,
            'TILE_HEIGHT': 8,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-256x4': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 256,
            'TILE_HEIGHT': 4,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-512x2': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 512,
            'TILE_HEIGHT': 2,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        'Block-1024x1': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 1024,
            'TILE_WIDTH': 1024,
            'TILE_HEIGHT': 1,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
    },
    # SET 3
    # Data Dimensions: 16384 x 16384 x 16384
    # Block Dimensions: 32 x 32 x 1
    # Computations per core: Variable
    'Set 3': {
        'Base': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 1,
            'PER_THREAD_Y': 1,
            'PER_THREAD_Z': 1,
        },
        '2Comps': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 2,
            'PER_THREAD_Y': 2,
            'PER_THREAD_Z': 2,
        },
        '4Comps': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 4,
            'PER_THREAD_Y': 4,
            'PER_THREAD_Z': 4,
        },
        '8Comps': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 8,
            'PER_THREAD_Y': 8,
            'PER_THREAD_Z': 8,
        },
        '16Comps': {
            'DIMENSIONS': 3,
            'ITERATIONS': 1024,
            'SIZE': 16384,
            'TILE_WIDTH': 32,
            'TILE_HEIGHT': 32,
            'TILE_DEPTH': 1,
            'PER_THREAD_X': 16,
            'PER_THREAD_Y': 16,
            'PER_THREAD_Z': 16,
        },
    }
}

if __name__ == '__main__':
    baseDir = os.path.dirname(os.path.realpath(__file__))
    blockedTemplate = os.path.join(baseDir, 'blocked_template.cu.template')
    naiveTemplate = os.path.join(baseDir, 'naive_template.cu.template')

    for setName, instances in cases:
        for name, case in instances:
            blockedTarget = os.path.join(baseDir, setName + '_' + name + '_blocked.cu')
            naiveTarget = os.path.join(baseDir, setName + '_' + name + '_naive.cu')

            replaceCases = {'%%' + k + '%%': v for k, v in case.iteritems()}
            with open(blockedTemplate, 'r') as template:
                with open(blockedTarget, 'w') as target:
                    for line in template:
                        target.write(line.replace(replaceCases))
