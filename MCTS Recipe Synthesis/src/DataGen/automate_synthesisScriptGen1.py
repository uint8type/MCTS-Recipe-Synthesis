import os
import sys
import random
import shutil
import argparse

homeDir = None
srcFolder = None
graphDataFolder = None
scriptsDataFolder = None
libraryCellFolder = None
designs = []
libraryName = None

numSynthesizedScript = 1500
delimiter = "\n"
def genSynthesisScripts():
    for i in range(numSynthesizedScript):
        srcFile = os.path.join(srcFolder, 'abc' + str(i) + '.script')

        if not os.path.exists(srcFile):
            print(f"[WARN] Source script not found: {srcFile}")
            continue

        with open(srcFile, 'r') as origScriptFile:
            fileLines = origScriptFile.readlines()

        for des in designs:
            scriptFolder = os.path.join(scriptsDataFolder, des)
            os.makedirs(scriptFolder, exist_ok=True)

            graphDumpFolder = os.path.join(graphDataFolder, des)
            os.makedirs(graphDumpFolder, exist_ok=True)

            scriptFilePath = os.path.join(scriptFolder, 'abc' + str(i) + '.script')

            with open(scriptFilePath, 'w+') as scriptFile:
                # Read library
                readLibLine = "read_lib " + os.path.join(libraryCellFolder, libraryName) + delimiter
                scriptFile.write(readLibLine)

                # Read design
                fileLines[1] = (
                    "read_bench "
                    + graphDumpFolder
                    + os.sep
                    + des
                    + "_orig.bench"
                    + delimiter
                )
                scriptFile.write(fileLines[1])

                scriptFile.write("strash" + delimiter)

                # Initial dump
                firstPathFileName = os.path.join(
                    graphDumpFolder,
                    "syn" + str(i),
                    des + "_syn" + str(i) + "_step0.bench" + delimiter,
                )
                os.makedirs(os.path.dirname(firstPathFileName), exist_ok=True)

                dumpFirstGraphLine = "write_bench -l " + firstPathFileName
                scriptFile.write(dumpFirstGraphLine)

                numSteps = 1
                for line in fileLines[2:-8]:
                    if line.startswith("write_bench"):
                        continue

                    scriptFile.write(line)

                    intermediatePathFileName = os.path.join(
                        graphDumpFolder,
                        "syn" + str(i),
                        des + "_syn" + str(i) + "_step" + str(numSteps) + ".bench" + delimiter,
                    )
                    numSteps += 1

                # Mapping + stats
                scriptFile.write("map" + delimiter)
                scriptFile.write("print_stats -p" + delimiter)

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, srcFolder, graphDataFolder
    global scriptsDataFolder, libraryCellFolder
    global designs, libraryName

    homeDir = cmdArgs.home
    srcFolder = cmdArgs.script
    designs = cmdArgs.designs
    libraryName = cmdArgs.lib

    if not (os.path.exists(homeDir) and os.path.exists(srcFolder)):
        raise RuntimeError("Invalid home directory or script folder path")

    graphDataFolder = os.path.join(homeDir, "OpenABC_DATA", "bench")
    scriptsDataFolder = os.path.join(homeDir, "OpenABC_DATA", "synScripts")
    libraryCellFolder = os.path.join(homeDir, "OpenABC_DATA", "lib")

    libPath = os.path.join(libraryCellFolder, libraryName)
    if not os.path.exists(libPath):
        raise RuntimeError(f"Library file not found: {libPath}")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(
        prog="SYNTHESIS RECIPE GENERATOR",
        description="Generate ABC synthesis scripts"
    )

    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('--home', required=True, help="OpenABC dataset home path")
    parser.add_argument('--script', required=True, help="Folder containing base abc*.script files")
    parser.add_argument('--lib', required=True, help="Library file name (e.g., sky130.lib)")
    parser.add_argument(
        '--designs',
        nargs='+',
        required=True,
        help="List of design names (e.g., c17 aes pci)"
    )

    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    genSynthesisScripts()

if __name__ == '__main__':
    main()
