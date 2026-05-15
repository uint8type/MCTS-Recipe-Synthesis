import os, sys,random,shutil
import argparse

homeDir = os.environ["HOME"]
graphDataFolder = os.path.join(homeDir,"OpenABC_DATA","bench")
scriptsDataFolder = os.path.join(homeDir,"OpenABC_DATA","synScripts")
libraryCellFolder = os.path.join(homeDir,"OpenABC_DATA","lib")


designs = []
numSynthesizedScript = 1500
delimiter = '\n'

def genShellScriptForSynthesis():
    for des in designs:
        designScriptFile = open(os.path.join(graphDataFolder,'synthesisBulk_'+des+'.sh'),'w+')
        print(os.path.join(graphDataFolder,'synthesisBulk_'+des+'.sh'))
        logFolder = os.path.join(graphDataFolder,des,'log_'+des)
        if not os.path.exists(logFolder):
            os.mkdir(logFolder)
        for i in range(numSynthesizedScript):
            synScriptPath = os.path.join(scriptsDataFolder,des,'abc'+str(i)+".script")
            logFilePath = os.path.join(logFolder,'log_'+des+'_syn'+str(i)+'.log')
            synRunCmd = 'yosys-abc -f '+synScriptPath+' > '+logFilePath
            synFolder = os.path.join(graphDataFolder,des,'syn'+str(i))
            if not os.path.exists(synFolder):
                os.mkdir(synFolder)
            zipCmd = 'zip -q -j -r '+synFolder+'.zip '+synFolder+"/"
            rmCmd = 'rm -fr '+synFolder+"/"
            designScriptFile.write(synRunCmd+delimiter)
            designScriptFile.write(zipCmd+delimiter)
            designScriptFile.write(rmCmd+delimiter)
        # zipCmd = 'zip -q -j -r '+synFolder+'.zip '+synFolder+"/"
        # rmCmd = 'rm -fr '+synFolder+"/"
        # designScriptFile.write(zipCmd+delimiter)
        # designScriptFile.write(rmCmd+delimiter)
        designScriptFile.close()

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, graphDataFolder,scriptsDataFolder,libraryCellFolder,designs
    homeDir = cmdArgs.home
    designs = cmdArgs.designs
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    graphDataFolder = os.path.join(homeDir,"OpenABC_DATA","bench")
    scriptsDataFolder = os.path.join(homeDir,"OpenABC_DATA","synScripts")
    libraryCellFolder = os.path.join(homeDir,"OpenABC_DATA","lib")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='AUTOMATE SYNTHESIS FLOW', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    parser.add_argument('--designs',nargs='+',required=True,help="List of design names:")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    genShellScriptForSynthesis()

if __name__ == '__main__':
    main()
