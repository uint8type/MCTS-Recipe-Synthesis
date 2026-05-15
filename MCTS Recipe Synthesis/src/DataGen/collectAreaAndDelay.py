import os,sys
import re,argparse
import os.path as osp

homeDir = None
benchDataFolder = None
statsDataFolder = None
designs = []
NUM_SYNTHESIZED_DESIGNS = 1500
csvDelimiter = ","
designSet = []

def getFileLines(filePath):
    f = open(filePath,'r')
    fLines = f.readlines()
    f.close()
    return fLines

def collectAreaAndDelay():
    adpFolder = osp.join(statsDataFolder,"adp")
    if not os.path.exists(adpFolder):
        os.mkdir(adpFolder)
    for des in designs:
        desLogDir = osp.join(benchDataFolder,des,"log_"+des)
        csv_file = os.path.join(adpFolder, 'adp_'+des+'.csv')
        csvFileHandler = open(csv_file,'w+')
        csvFileHandler.write("sid,power\n")
        for i in range(NUM_SYNTHESIZED_DESIGNS):
            synth_stat_file = os.path.join(desLogDir,'log_'+des+"_syn"+str(i)+'.log')
            synthFileLines = getFileLines(synth_stat_file)
            information = re.findall('[a-zA-Z0-9.]+',synthFileLines[-1])
            #power_value = information[-1]  # Assuming power is the last value in the log line
            #csvFileHandler.write(str(i)+csvDelimiter+str(power_value)+"\n")
            csvFileHandler.write(str(i)+csvDelimiter+str(information[-1])+"\n")
            #csvFileHandler.write(str(i)+csvDelimiter+str(information[-7])+csvDelimiter+str(information[-5])+"\n")      
        csvFileHandler.close()

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir,benchDataFolder,statsDataFolder,designs
    homeDir = cmdArgs.home
    designs = cmdArgs.designs
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    benchDataFolder = os.path.join(homeDir,"OpenABC_DATA","bench")
    statsDataFolder = os.path.join(homeDir,"OpenABC_DATA","statistics")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='Final AIG area and delay Collection', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    parser.add_argument('--designs',nargs='+',required=True,help="List of design names:")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    collectAreaAndDelay()
    print("Done!")


if __name__ == '__main__':
    main()
