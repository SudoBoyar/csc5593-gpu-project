import os
import sys

from cases import cases

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
    if len(sys.argv >= 1):
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
