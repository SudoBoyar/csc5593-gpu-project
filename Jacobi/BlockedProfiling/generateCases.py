import os
import sys

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
    'Set1': {
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
    'Set3': {
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

makefileTemplate = """
BINARIES={binaries}

all: $(BINARIES)

{cases}
"""

heraclesMakefileCaseTemplate = """
{name}: {file}
	ssh node18 nvcc -std=c++11 {file} -o {outFile}
"""

if __name__ == '__main__':
    if count(sys.argv >= 1):
        platform = sys.argv[1]
    else:
        platform = 'heracles'

    if platform not in ['heracles', 'dozer']:
        print 'Only heracles or dozer'
        exit(1)

    baseDir = os.path.dirname(os.path.realpath(__file__))
    blockedTemplate = os.path.join(baseDir, 'blocked_template.cu.template')
    naiveTemplate = os.path.join(baseDir, 'naive_template.cu.template')

    binaries = []
    makefileCases = ""

    for setName, instances in cases.iteritems():
        for caseName, case in instances.iteritems():
            name = setName + '_' + caseName
            blockedTarget = os.path.join(baseDir, name + '_blocked.cu')
            naiveTarget = os.path.join(baseDir, name + '_naive.cu')
            blockedOutfile = os.path.join(baseDir, name + '_blocked')
            naiveOutfile = os.path.join(baseDir, name + '_naive')

            if platform == 'heracles':
                makefileCases += heraclesMakefileCaseTemplate.format(**{'name': name, 'file': blockedTarget, 'outFile': blockedOutfile})
                makefileCases += heraclesMakefileCaseTemplate.format(**{'name': name, 'file': naiveTarget, 'outFile': naiveOutfile})

            binaries.append(name)

            replaceCases = {'%%' + k + '%%': v for k, v in case.iteritems()}
            with open(blockedTemplate, 'r') as template:
                with open(blockedTarget, 'w') as target:
                    for line in template:
                        line = line.strip()
                        for find, replace in replaceCases.iteritems():
                            line = line.replace(find, str(replace))
                        target.write(line)

    makefileContents = makefileTemplate.format(**{'binaries': str.join(' ', binaries), 'cases': makefileCases})
    with open(os.path.join(baseDir, 'Makefile'), 'w') as makefile:
        makefile.write(makefileContents)
