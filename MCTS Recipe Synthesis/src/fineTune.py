# nohup python fine_tune.py --datadir $HOME/OPENABC_DATASET --rundir $HOME/OpenABC/models/qor/SynthNetV3/NETV3_set2 --ft 0 --lr 0.001 --epochs 20 --batch_size 8 --design bp_be --budget 50 --model gcn-epoch-57-val_loss-0.490.pt > ft_bp_be_standalone_out.log 2> ft_bp_be_standalone_err.log &

import os

import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
from utils import *
from netlistDataset import *

import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
import os.path as osp
import pickle
import sys
from sklearn.cluster import KMeans
import random
import csv


DUMP_DIR = None
criterion = torch.nn.MSELoss()
design = None

def plotChart(x,y,xlabel,ylabel,leg_label,title):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, label=leg_label)
    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title,weight='bold')
    plt.savefig(osp.join(title+'.png'), format='png', bbox_inches='tight')

def train(model,device,dataloader,optimizer):
    torch.cuda.empty_cache()
    epochLoss = AverageMeter()
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
        batch = batch.to(device)
        lbl = batch.target.reshape(-1, 1)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred,lbl)
        loss.backward()
        optimizer.step()
        numInputs = pred.view(-1,1).size(0)
        epochLoss.update(loss.detach().item(),numInputs)
        torch.cuda.empty_cache()
    return epochLoss.avg


def evaluate(model, device, dataloader):
    torch.cuda.empty_cache()
    model.eval()
    validLoss = AverageMeter()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(device)
            pred = model(batch)
            lbl = batch.target.reshape(-1, 1)
            mseVal = mse(pred, lbl)
            numInputs = pred.view(-1,1).size(0)
            validLoss.update(mseVal,numInputs)
            torch.cuda.empty_cache()
    return validLoss.avg

def evaluate_plot(model, device, dataloader):
    torch.cuda.empty_cache()
    model.eval()
    totalMSE = AverageMeter()
    batchData = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(device)
            pred = model(batch)
            lbl = batch.target.reshape(-1, 1)
            desName = batch.desName
            synID = batch.synID
            predArray = pred.view(-1,1).detach().cpu().numpy()
            actualArray = lbl.view(-1,1).detach().cpu().numpy()
            batchData.append([predArray,actualArray,desName,synID])
            mseVal = mse(pred, lbl)
            numInputs = pred.view(-1,1).size(0)
            totalMSE.update(mseVal,numInputs)

            torch.cuda.empty_cache()

    return totalMSE.avg,batchData



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on Synthesis Task Pytorch Geometric')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--ft', type=int, default=2,
                        help='Fine Tuning Mode (Standalone:0, FT+R: 1, FT+A: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--rundir', type=str, required=True,default="",
                        help='Output directory path to store result')
    parser.add_argument('--datadir', type=str, required=True, default="",
                        help='Dataset directory containing processed dataset, train test split file csvs')
    parser.add_argument('--target', type=str, required=False, default="nodes",
                        help='Target label (nodes/area/delay), default:"nodes"')
    parser.add_argument('--model', type=str, required=True, default="",
                        help='Pre-trained model name in path <rundir> (eg. gcn-epoch30-loss-0.7734.pt)')
    parser.add_argument('--design', type=str, required=True, default="",
                        help='Design Name (eg. aes_secworks)')
    parser.add_argument('--budget', type=int, required=False, default=30,
                        help='Synthesis Budget -- Number of actual synthesis runs')
    parser.add_argument('--stats_file', type=str, required=True, help="Path to the synthesis statistics pickle file")
    
    args = parser.parse_args()
    #RUN_DIR = args.rundir

    # Hyperparameters
    batchSize = args.batch_size #64
    num_epochs = args.epochs #80
    learning_rate = args.lr #0.001
    ft_mode = args.ft
    targetLbl = args.target
    nodeEmbeddingDim = 3
    synthEncodingDim = 3
    MODEL_NAME = args.model
    design = args.design
    budget = args.budget

    IS_STATS_AVAILABLE = True
    ROOT_DIR = args.datadir 
    global DUMP_DIR
    DUMP_DIR = args.rundir 
    
    if ft_mode == 1:
        dir_name = "Random"
    elif ft_mode == 2:
        dir_name = "Active" 
    elif ft_mode == 0:
        dir_name = "Standalone"

    if not osp.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)

    NEW_DUMP_DIR = DUMP_DIR+"/"+design+"/"+dir_name

    if not osp.exists(DUMP_DIR+"/"+design):
        os.mkdir(DUMP_DIR+"/"+design)
    
    if not osp.exists(NEW_DUMP_DIR):
        os.mkdir(NEW_DUMP_DIR)

    MODEL_PATH = osp.join(DUMP_DIR,MODEL_NAME)

    if ft_mode == 1:
        ft_recipes = random.sample(range(1500), budget)
    elif ft_mode == 2 or ft_mode == 0:
        with open('syn_recipe_embeddings.pickle', 'rb') as file:
            dict = pickle.load(file)

        feature_vectors = list(dict.values())
        feature_matrix = np.array(feature_vectors)
        kmeans = KMeans(n_clusters=budget, random_state=42)
        kmeans.fit(feature_matrix)

        cluster_centers = kmeans.cluster_centers_
        cluster_indices = [np.argmin(np.linalg.norm(feature_matrix - center, axis=1)) for center in cluster_centers]

        keys_of_cluster_heads = list(dict.keys())
        ft_recipes = [keys_of_cluster_heads[i] for i in cluster_indices]

    csv_train_data = [['fileName']]
    csv_test_data = [['fileName']]
    for rec in ft_recipes:
        row = []
        row.append(design+"_syn"+str(rec)+"_step0.pt.zip")
        csv_train_data.append(row)
    
    for rec in range(0,1500):
        row = []
        row.append(design+"_syn"+str(rec)+"_step0.pt.zip")
        csv_test_data.append(row)

    with open(osp.join(ROOT_DIR, "lp1/"+design+"_"+dir_name.lower()+"_train.csv"), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_train_data)
    
    with open(osp.join(ROOT_DIR, "lp1/"+design+"_"+dir_name.lower()+"_test.csv"), 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_test_data)
    
    
    # Load train and test datasets
    # Load train and test datasets
    trainDS = NetlistGraphDataset(root=osp.join(ROOT_DIR, "lp1"),
                                  filePath=design+"_"+dir_name.lower()+"_train.csv")
    testDS = NetlistGraphDataset(root=osp.join(ROOT_DIR, "lp1"),
                                  filePath=design+"_"+dir_name.lower()+"_test.csv")
    if IS_STATS_AVAILABLE:
        # with open(osp.join(ROOT_DIR,'statistics','synthesisStatistics12.pickle'),'rb') as f:
        #     targetStats = pickle.load(f)
        stats_path = args.stats_file
        with open(stats_path, 'rb') as f:
            targetStats = pickle.load(f)

    else:
        print("\nNo pickle file found for number of gates")
        exit(0)

    meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStats,targetVar=targetLbl)

    trainDS.transform = transforms.Compose([lambda data: addAbsoluteTargets(data,targetStats,targetVar=targetLbl)])
    testDS.transform = transforms.Compose([lambda data: addAbsoluteTargets(data,targetStats,targetVar=targetLbl)])

    num_classes = 1


    # Define the model
    synthFlowEncodingDim = trainDS[0].synVec.size()[0]*synthEncodingDim
    node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
    synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim)

    model = SynthNet(node_encoder=node_encoder,synth_encoder=synthesis_encoder,n_classes=num_classes,synth_input_dim=synthFlowEncodingDim,node_input_dim=nodeEmbeddingDim+2)
    if ft_mode == 1 or ft_mode == 2:
        model.load_state_dict(torch.load(MODEL_PATH))
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    device = getDevice()
    model = model.to(device)

    # Split the training data into training and validation dataset
    training_validation_samples = [int(0.8*len(trainDS)),len(trainDS)-int(0.8*len(trainDS))]
    train_DS,valid_DS = random_split(trainDS,training_validation_samples)


    # Initialize the dataloaders
    train_dl = DataLoader(train_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
    valid_dl = DataLoader(valid_DS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)
    test_dl = DataLoader(testDS,shuffle=True,batch_size=batchSize,pin_memory=True,num_workers=4)


    # Monitor the loss parameters
    valid_curve = []
    train_loss = []
    validLossOpt = 0
    bestValEpoch = 1


    for ep in range(1, num_epochs + 1):
        print("\nEpoch [{}/{}]".format(ep, num_epochs))
        print("\nTraining..")
        trainLoss = train(model, device, train_dl, optimizer)
        print("\nEvaluation..")
        validLoss = evaluate(model, device, valid_dl)
        if ep > 1:
            if validLossOpt > validLoss:
                validLossOpt = validLoss
                bestValEpoch = ep
                torch.save(model.state_dict(), osp.join(NEW_DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt)))
        else:
            validLossOpt = validLoss
            torch.save(model.state_dict(), osp.join(NEW_DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt)))
        print({'Train loss': trainLoss,'Validation loss': validLoss})
        valid_curve.append(validLoss)
        train_loss.append(trainLoss)
        scheduler.step(validLoss)

    # Loading best validation model
    model.load_state_dict(torch.load(osp.join(NEW_DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestValEpoch, validLossOpt))))


    # Save training data for future plots
    with open(osp.join(NEW_DUMP_DIR,'valid_curve.pkl'),'wb') as f:
        pickle.dump(valid_curve,f)

    with open(osp.join(NEW_DUMP_DIR,'train_loss.pkl'),'wb') as f:
        pickle.dump(train_loss,f)

    plotChart([i+1 for i in range(len(valid_curve))],valid_curve,"# Epochs","Loss","test_acc", NEW_DUMP_DIR+"/Validation Loss")
    plotChart([i+1 for i in range(len(train_loss))],train_loss,"# Epochs","Loss","train_loss","/"+NEW_DUMP_DIR+"/Training Loss")

    # # Evaluate on train data
    # trainMSE,trainBatchData = evaluate_plot(model, device, train_dl)
    # NUM_BATCHES_TRAIN = len(train_dl)
    # doScatterAndTopKRanking(NUM_BATCHES_TRAIN,batchSize,trainBatchData,DUMP_DIR,"train")

    # # Evaluate on validation data
    # validMSE,validBatchData = evaluate_plot(model, device, valid_dl)
    # NUM_BATCHES_VALID = len(valid_dl)
    # doScatterAndTopKRanking(NUM_BATCHES_VALID,batchSize,validBatchData,DUMP_DIR,"valid")

    # Evaluate on test data
    testMSE,testBatchData = evaluate_plot(model, device, test_dl)
    NUM_BATCHES_TEST = len(test_dl)
    doScatterAndTopKRanking(NUM_BATCHES_TEST,batchSize,testBatchData,NEW_DUMP_DIR,"test")
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print("Final run statistics")
    print(f'Total Params: {num_params}')
    # print("Training loss per sample:{}".format(trainMSE))
    # print("Validation loss per sample:{}".format(validMSE))
    print("Test loss per sample:{}".format(testMSE))

if __name__ == "__main__":
    main()
