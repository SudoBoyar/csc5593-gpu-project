import os
import shutil
import sys

from cases import cases

makefileTemplate = """
BINARIES={binaries}

all: $(BINARIES)

{cases}
"""

heraclesMakefileCaseTemplate = """
{name}: {file}
	ssh node18 nvcc {file} -o {outFile}
"""

bashRunFileTemplate = """
#!/bin/bash
echo "{name}"
{commands}
"""

heraclesRunTemplate = """
echo "{name}"
ssh node18 {outPath}
"""

def usage():
    print "python generateCases.py [option] [setIndex [setIndex...]]"
    print "\tOptions:"
    print "\t\thelp         print this message"
    print "\t\tclean        remove the generated directory"
    print "\t\theracles     build for heracles"
    print "\t\tdozer        build for dozer"
    print "\t\thydra        build for hydra"
    print "\n"
    print "\tExample:"
    print "\t\tTo generate full suite on heracles"
    print "\t\t$ python generateCases.py heracles"
    print "\t\tOr to only generate the 1st and 3rd sets"
    print "\t\t$ python generateCases.py heracles 1 3"

def baseDirectory():
    return os.path.dirname(os.path.realpath(__file__))

def generatedDirectory():
    return os.path.join(baseDirectory(), 'generated')

def writeFileWithReplacements(replaceCases, templateFile, targetFile):
    with open(templateFile, 'r') as template:
        with open(targetFile, 'w') as target:
            for line in template:
                for find, replace in replaceCases.iteritems():
                    line = line.replace(find, str(replace))
                target.write(line)


def parseArgs():
    platform = None
    setNums = []

    if len(sys.argv) >= 2:
        platform = sys.argv[1]
    if len(sys.argv) >= 2:
        setNums = sys.argv[2:]

    if platform == 'help':
        usage()
        sys.exit(0)

    elif platform == 'clean':
        if os.path.exists(generatedDirectory()):
            shutil.rmtree(generatedDirectory())

        print "Cleaned\n"
        sys.exit(0)

    elif platform is None:
        usage()
        sys.exit(1)

    elif platform not in ['heracles', 'dozer', 'hydra']:
        print "I don't know what \"{platform}\" means\n".format(platform = platform)
        usage()
        sys.exit(1)

    return platform, setNums

if __name__ == '__main__':
    platform, setNums = parseArgs()

    binaries = []
    makefileCases = ""

    if not os.path.exists(generatedDirectory()):
        os.mkdir(generatedDirectory())

    for setName, setData in cases.iteritems():
        if setData.get('disable', False):
            continue

        if len(setNums) > 0 and setName not in ['Set' + str(setNum) for setNum in setNums]:
            continue

        files = setData['files']
        specs = setData['specs']

        setDir = os.path.join(generatedDirectory(), setName)
        if not os.path.exists(setDir):
            os.mkdir(setDir)

        setRunFile = os.path.join(setDir, 'run' + setName + '.sh')
        setRunCommands = ''

        for caseName, case in specs.iteritems():
            name = caseName

            caseRunFile = os.path.join(setDir, 'run' + name + '.sh')
            caseRunCommands = ''

            for fileData in files:
                binary = name + fileData['out_suffix']
                target = os.path.join(setDir, name + fileData['cu_suffix'])
                outFile = os.path.join(setDir, binary)

                if platform == 'heracles':
                    makefileCases += heraclesMakefileCaseTemplate.format(
                        **{'name': binary, 'file': target, 'outFile': outFile})

                    setRunCommands += heraclesRunTemplate.format(name=binary, outPath = outFile)
                    caseRunCommands += heraclesRunTemplate.format(name=binary, outPath = outFile)

                binaries.append(binary)

                replaceCases = {'%%' + k + '%%': v for k, v in case.iteritems()}
                templatePath = os.path.join(baseDirectory(), fileData['template'])
                writeFileWithReplacements(replaceCases, templatePath, target)
                # if (os.path.exists(target)):
                #     tmpFile = os.path.join(setDir, 'tmp.cu')
                #     writeFileWithReplacements(replaceCases, templatePath, tmpFile)
                # else:
                #     writeFileWithReplacements(replaceCases, templatePath, target)

            runscriptContents = bashRunFileTemplate.format(commands=caseRunCommands, name=name)
            with open(caseRunFile, 'w') as runscript:
                runscript.write(runscriptContents)
                os.fchmod(runscript.fileno(), 744)

        runscriptContents = bashRunFileTemplate.format(commands=setRunCommands, name=setName)
        with open(setRunFile, 'w') as runscript:
            runscript.write(runscriptContents)
            os.fchmod(runscript.fileno(), 744)

    makefileContents = makefileTemplate.format(**{'binaries': str.join(' ', binaries), 'cases': makefileCases})
    with open(os.path.join(baseDirectory(), 'Makefile'), 'w') as makefile:
        makefile.write(makefileContents)
