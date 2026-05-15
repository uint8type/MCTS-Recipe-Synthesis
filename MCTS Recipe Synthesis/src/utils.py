from statistics import mean
from webbrowser import get
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import numpy as np

def getMeanAndVariance(targetList):
    return np.mean(np.array(targetList)),np.std(np.array(targetList))

def computeMeanAndVarianceOfTargets(targetStatsDict,targetVar='power'):
    meanAndVarTargetDict = {}
    # print(targetStatsDict.keys())
    for des in targetStatsDict.keys():
        numNodes,_,_,powerVar = targetStatsDict[des]
        if targetVar == 'power':
            meanTarget,varTarget = getMeanAndVariance(powerVar)
        '''
        if targetVar == 'power':
            meanTarget,varTarget = getMeanAndVariance(powerVar)
        elif targetVar == 'power':
            meanTarget,varTarget = getMeanAndVariance(powerVar)
        else:
            meanTarget,varTarget = getMeanAndVariance(numNodes)
        '''
        meanAndVarTargetDict[des] = [meanTarget,varTarget]
    return meanAndVarTargetDict

def addNormalizedTargets(data,targetStatsDict,meanVarDataDict,targetVar='power'):
    sid = data.synID[0]
    desName = data.desName[0]
    if targetVar == 'power':    
        targetIdentifier = 3 # Column number of target 'power' in synthesisStatistics.pickle entries
        #normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        normTarget = targetStatsDict[desName][targetIdentifier][sid]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    '''
    if targetVar == 'power':    
        targetIdentifier = 4 # Column number of target 'power' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    elif targetVar == 'power':
        targetIdentifier = 3 # Column number of target 'power' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    else:
        targetIdentifier = 0 # Column number of target 'Nodes' in synthesisStatistics.pickle entries
        normTarget = (targetStatsDict[desName][targetIdentifier][sid] - meanVarDataDict[desName][0]) / meanVarDataDict[desName][1]
        data.target = torch.tensor([normTarget],dtype=torch.float32)
    '''
    return data


def addAbsoluteTargets(data,targetStatsDict,targetVar='power'):
    sid = data.synID[0]
    desName = data.desName[0]
    if desName.endswith('_copy'):
        desName = desName[:-5]
    numNodes,_,_,powerVar = targetStatsDict[desName]
    if targetVar == 'power':
        data.target = torch.tensor([powerVar[sid]],dtype=torch.float32)
    '''
    if targetVar == 'power':
        data.target = torch.tensor([powerVar[sid]],dtype=torch.float32)
    elif targetVar == 'power':
        data.target = torch.tensor([powerVar[sid]],dtype=torch.float32)
    else:
        data.target = torch.tensor([numNodes[sid]],dtype=torch.float32)'
    '''
    return data

# Torch.std_mean returns tuple with std first and mean second term
def mapMeanChangeToTensor(data,powerStatsDict):
    power = data.power
    #power = data.power
    data.power = (power - powerStatsDict[data.desName[0]][1]) / powerStatsDict[data.desName[0]][0]
    #data.power = (power - powerStatsDict[data.desName[0]][1]) / powerStatsDict[data.desName[0]][0]
    assert(data.power > -10 and data.power < 10)
    return data

# Element 0 is power and 1 is power
def getMeanpowerAndpower(trainDS,testDS):
    desNamesTrain = set(elem.desName[0] for elem in trainDS)
    desNamesTest = set(elem.desName[0] for elem in testDS)
    desNameTotal = desNamesTrain.union(desNamesTest)
    #desStatspower = {}
    desStatspower = {}
    powerStats = {}
    for des in desNameTotal:
        desStatspower[des] = []
        #desStatspower[des] = []
    for elem in trainDS:
        #desStatspower[elem.desName[0]].append(elem.power)
        desStatspower[elem.desName[0]].append(elem.power)
        #desStatspower[elem.desName[0]].append(elem.power)
    for elem in testDS:
        #desStatspower[elem.desName[0]].append(elem.power)
        desStatspower[elem.desName[0]].append(elem.power)
        #desStatspower[elem.desName[0]].append(elem.power)
    for des in desNameTotal:
        #powerStats[des] = torch.std_mean(torch.tensor(desStatspower[des]))
        powerStats[des] = torch.std_mean(torch.tensor(desStatspower[des]))
        #powerStats[des] = torch.std_mean(torch.tensor(desStatspower[des]))
    return powerStats


def getMinMaxTargetVal(dataSet):
    desMinMaxpowerVal = {}
    #desMinMaxpowerVal = {}
    desNames = [elem.desName[0] for elem in dataSet]
    for des in desNames:
        desMinMaxpowerVal[des] = [None,None]
        #desMinMaxpowerVal[des] = [None,None]
    for ditem in dataSet[1:]:
        des = ditem.desName[0]
        #power = ditem.power
        power = ditem.power
        # power computation
        desMinMaxpowerVal[des][0] = power if (power > desMinMaxpowerVal[des][0] or desMinMaxpowerVal[des][0] == None) else desMinMaxpowerVal[des][0]
        desMinMaxpowerVal[des][1] = power if (power < desMinMaxpowerVal[des][1] or desMinMaxpowerVal[des][1] == None) else desMinMaxpowerVal[des][1]
        # power computation
        #desMinMaxpowerVal[des][0] = power if (power > desMinMaxpowerVal[des][0] or desMinMaxpowerVal[des][1] == None) else desMinMaxpowerVal[des][0]
        #desMinMaxpowerVal[des][1] = power if (power < desMinMaxpowerVal[des][1] or desMinMaxpowerVal[des][1] == None) else desMinMaxpowerVal[des][1]
    return desMinMaxpowerVal

def checkUnseenDesInTest(powerDict,testDS):
    unseenDesigns = set(elem.desName[0] for elem in testDS if not elem.desName[0] in powerDict.keys())
    if len(unseenDesigns) > 0:
        desMinMaxpowerVal = {}
        #desMinMaxpowerVal = {}
        for des in unseenDesigns:
            desMinMaxpowerVal[des] = [0, -1]
            #desMinMaxpowerVal[des] = [0,-1]
        for ditem in testDS:
            des = ditem.desName[0]
            power = ditem.power
            #power = ditem.power
            if( not des in unseenDesigns):
                pass
            # power computation
            desMinMaxpowerVal[des][0] = power if power > desMinMaxpowerVal[des][0] else desMinMaxpowerVal[des][0]
            desMinMaxpowerVal[des][1] = power if (power < desMinMaxpowerVal[des][1] or power == -1) else desMinMaxpowerVal[des][1]
            # power computation
            #desMinMaxpowerVal[des][0] = power if power > desMinMaxpowerVal[des][0] else desMinMaxpowerVal[des][0]
            #desMinMaxpowerVal[des][1] = power if (power < desMinMaxpowerVal[des][1] or power == -1) else desMinMaxpowerVal[des][1]
        return desMinMaxpowerVal
    else:
        return None,None


def getDevice():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def desName_to_idx(aigData):
    desNames = [elem.desName[0] for elem in aigData]
    desNameIdxDict = {}
    idxDesNameDict = {}
    i=0
    for des in desNames:
        if not des in desNameIdxDict.keys():
            desNameIdxDict[des] = i
            idxDesNameDict[i] = des
            i+=1
    return desNameIdxDict,idxDesNameDict

def mapNameToLabel(data,desNameIdxDict):
    labelName = data.desName[0]
    data.desLabel = torch.tensor([desNameIdxDict[labelName]])
    return data

def mapAttributesToTensor(data,powerDict):
    power = data.power
    #power = data.power
    minMaxpower = powerDict[data.desName[0]]
    #minMaxpower = powerDict[data.desName[0]]
    data.power = (power - minMaxpower[1])/(minMaxpower[0] - minMaxpower[1])
   # data.power = (power - minMaxpower[1]) / (minMaxpower[0] - minMaxpower[1])
    return data


def mse(y_pred,y_true):
    return mean_squared_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def mae(y_pred,y_true):
    return mean_absolute_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def doScatterPlot(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".png")
        #else:
        #    fileName = osp.join(dumpDir,"scatterPlot_test_"+d+".png")
        plt.savefig(fileName,bbox_inches='tight')


def getTopKSimilarityPercentage(list1,list2,topkpercent):
    listLen = len(list1)
    topKIndexSimilarity = int(topkpercent*listLen)
    Set1 = set(list1[:topKIndexSimilarity])
    Set2 = set(list2[:topKIndexSimilarity])
    numSimilarScripts = len(Set1.intersection(Set2))
    if topKIndexSimilarity >0:
        return (numSimilarScripts/topKIndexSimilarity)
    else:
        return 0


def doScatterAndTopKRanking(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    synthesisID = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])
            synthesisID.append(batchData[i][3][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'synID': synthesisID,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    accuracyFile = osp.join(dumpDir, "topKaccuracy_" + trainMode + ".csv")
    accuracyFileWriter = open(accuracyFile,'w+')
    accuracyFileWriter.write("design,top1,top5,top10,top15,top20,top25"+"\n")
    endDelim = "\n"
    commaDelim = ","

    print("\nDataset type: "+trainMode)
    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d,weight='bold',fontsize=25)
        plt.xlabel('Actual', weight='bold', fontsize=25)
        plt.ylabel('Predicted', weight='bold', fontsize=25)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".png")
        plt.savefig(fileName,bbox_inches='tight')
        desDF1 = designDF.sort_values(by=['actual'])
        desDF2 = designDF.sort_values(by=['prediction'])
        desDF1_synID = desDF1.synID.to_list()
        desDF2_synID = desDF2.synID.to_list()
        kPercentSimilarity = [0.01,0.05,0.1,0.15,0.2,0.25]
        accuracyFileWriter.write(d)
        for kPer in kPercentSimilarity:
            topKPercentSimilarity = getTopKSimilarityPercentage(desDF1_synID,desDF2_synID,kPer)
            accuracyFileWriter.write(commaDelim+str(topKPercentSimilarity))
        accuracyFileWriter.write(endDelim)
        desDF1.to_csv(osp.join(dumpDir,"desDF1_"+trainMode+"_"+d+".csv"),index=False)
        desDF2.to_csv(osp.join(dumpDir,"desDF2_"+trainMode+"_"+d+".csv"),index=False)
        mapeScore = mean_absolute_percentage_error(designDF.prediction.to_list(),designDF.actual.to_list())
        print("MAPE ("+d+"): "+str(mapeScore))
    
    accuracyFileWriter.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count