#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data")
args = parser.parse_args()

data_dir = args.data

start = time()
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

if torch.cuda.is_available() == True:
    device = "cuda"
    print("Use GPU!")
else:
    device = "cpu"
    
norm = True
testdata = np.load(data_dir)

if norm:
    print("preprocessing data")
    scalerA = StandardScaler()
    scalerA.fit(testdata[:,0,:])
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    scalerB = StandardScaler()
    scalerB.fit(testdata[:,1,:])
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp
    
class RawDataset(Dataset):
    def __init__(self, traindata, trainlabel):
        self.traindata = torch.from_numpy(np.array(traindata).astype(np.float32))
        if trainlabel is not None:
            self.trainlabel = torch.from_numpy(np.array(trainlabel).astype(np.int64))
        else:
            self.trainlabel = None
    def __len__(self):
        return len(self.traindata)
    def __getitem__(self,idx):
        sample = self.traindata[idx]
        if self.trainlabel is not None:
            target = self.trainlabel[idx]
            return sample, target
        else:
            return sample
        
testset = RawDataset(testdata,None) 
testloader = DataLoader(testset,batch_size=1,shuffle=False)

class MyDSPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define your arch
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size=13, stride=7), #2285 * 20
            nn.ReLU(),
            nn.Conv1d(20, 40, kernel_size=11, stride=7), #326 * 40
            nn.ReLU(),
            nn.Conv1d(40, 80, kernel_size=9, stride=5), #64 * 80
            nn.ReLU(),
            nn.Conv1d(80, 160, kernel_size=7, stride=5), #12 * 160
        )
        self.clf = nn.Linear(1920, 3)
    def forward(self, x):
        # define your forward
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        output = self.clf(x)
        return output
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss
    
model = MyDSPNet()
model_path = './weight.pt'
model.load_state_dict(torch.load(model_path))

model = model.to(device)
with open("result.csv","w") as f:
    f.write("id,category\n")
    for i, x in enumerate(testloader):
        x = x.to(device)
        output = model(x)
        pred = output.argmax(dim=1, keepdim=True)
        f.write("%d,%d\n"%(i,pred.item()))

    
    
    
    
    
    