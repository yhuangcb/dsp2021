#%%
from re import L
import numpy as np
import torch, os, random
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl

#%%
if torch.cuda.is_available() == True:
    device = "cuda"
    print("Use GPU!")
else:
    device = "cpu"
# %%
norm = True
traindata, trainlabel = np.load("traindata.npy"), np.load("trainlabel.npy")
anomaly_sample = np.load("anomaly_sample.npy")
testdata = np.load("anomalytestdata.npy")
# %%
if norm:
    print("preprocessing data")
    scalerA = StandardScaler()
    scalerA.fit(traindata[:,0,:])
    tmp = scalerA.transform(traindata[:,0,:])
    traindata[:,0,:] = tmp
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    scalerB = StandardScaler()
    scalerB.fit(traindata[:,1,:])
    tmp = scalerB.transform(traindata[:,1,:])
    traindata[:,1,:] = tmp
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp

    tmp = scalerA.transform(anomaly_sample[:,0,:])
    anomaly_sample[:,0,:] = tmp
    tmp = scalerB.transform(anomaly_sample[:,1,:])
    anomaly_sample[:,1,:] = tmp
# %%
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
# %%
trainset = RawDataset(traindata, trainlabel) 
trainloader = DataLoader(trainset,batch_size=48,shuffle=True)
testset = RawDataset(testdata,None) 
testloader = DataLoader(testset,batch_size=1,shuffle=False)
anomalyset = RawDataset(anomaly_sample, None)
anomalyloader = DataLoader(anomalyset,batch_size=1,shuffle=False)
# %%
class VAE(nn.Module):
    def __init__(self, z_dim, x_dim=2*16000):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encode_conv = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size=13, stride=7), #2285 * 20
            nn.ReLU(),
            nn.Conv1d(20, 40, kernel_size=11, stride=7), #326 * 40
            nn.ReLU(),
            nn.Conv1d(40, 80, kernel_size=9, stride=5), #64 * 80
            nn.ReLU(),
            nn.Conv1d(80, 160, kernel_size=7, stride=5), #12 * 1
            nn.ReLU(),
            nn.Conv1d(160, 1, kernel_size=1, stride=1),
        )
        self.fc_mean = nn.Linear(12, z_dim)
        self.fc_var = nn.Linear(12, z_dim)
        nn.init.zeros_(self.fc_var.weight)
        self.decode_conv = nn.Sequential(
            nn.ConvTranspose1d(1, 160, kernel_size=1, stride=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(160, 80, kernel_size=7, stride=5, output_padding=2),
            nn.ReLU(), #64 * 80
            nn.ConvTranspose1d(80, 40, kernel_size=9, stride=5, output_padding=2),
            nn.ReLU(), #326 * 40
            nn.ConvTranspose1d(40, 20, kernel_size=11, stride=7, padding=1, output_padding=1),
            nn.ReLU(), #2285, 20
            nn.ConvTranspose1d(20, 2, kernel_size=13, stride=7, padding=1, output_padding=1),
            #16000, 2
        )

        self.encode_fc = nn.Sequential(
            #nn.Linear(32000, 2000),
            #nn.ReLU(),
            #nn.Linear(8000, 2000),
            #nn.ReLU(),
            #nn.Linear(2000, 200),
            #nn.ReLU(),
            nn.Linear(32000, z_dim),
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(z_dim, 32000),
            #nn.ReLU(),
            #nn.Linear(200, 2000),
            #nn.ReLU(),
            #nn.Linear(2000, 8000),
            #nn.ReLU(),
            #nn.Linear(2000, 32000),
        )

    def encoder(self, x):
        x = self.encode_conv(x)
        x = x.view(x.size(0), -1)
        #x = self.encode_fc(x)
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var 

    def reparameterization(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        #print(epsilon)
        return mean + epsilon*torch.exp(0.5*log_var)

    def decoder(self, z):
        #y = self.decode_conv(z)
        z = z.view(z.size(0), 1, -1)
        y = self.decode_fc(z)
        return y

    def __norm(self, data):
        scale = data.max() - data.min() + 1e-8
        norm_data = (data-data.min())/scale
        return norm_data

    def forward(self, x, device):
        #x = x.view(x.size(0), -1)
        mean, log_var = self.encoder(x)
        KL = 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var))
        z = self.reparameterization(mean, log_var, device)
        x_hat = self.decoder(z) 
        x = self.__norm(x)
        x_hat = self.__norm(x_hat)

        x = x.view(x.size(0), -1)
        x_hat = x_hat.view(x_hat.size(0), -1)
        recon = torch.sum(x * torch.log(x_hat+1e-8) + (1 - x) * torch.log(1 - x_hat  + 1e-8))
        lower_bound = -(KL + recon)
        #BCE = F.binary_cross_entropy(x_hat+1e-8, x+1e-8, reduction='sum')
        #lower_bound = BCE - KL
        #print(KL)
        #print(recon)

        return lower_bound, z, x_hat


#%%
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size=13, stride=7), #2285 * 20
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Conv1d(20, 40, kernel_size=11, stride=7), #326 * 40
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Conv1d(40, 80, kernel_size=9, stride=5), #64 * 80
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Conv1d(80, 160, kernel_size=7, stride=5), #12 * 1
            #nn.ReLU(),
            #nn.Conv1d(160, 1, kernel_size=1, stride=1),
        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose1d(1, 160, kernel_size=1, stride=1, output_padding=0),
            #nn.ReLU(),
            nn.ConvTranspose1d(160, 80, kernel_size=7, stride=5, output_padding=2),
            nn.BatchNorm1d(80),
            nn.ReLU(), #64 * 80
            nn.ConvTranspose1d(80, 40, kernel_size=9, stride=5, output_padding=2),
            nn.BatchNorm1d(40),
            nn.ReLU(), #326 * 40
            nn.ConvTranspose1d(40, 20, kernel_size=11, stride=7, padding=1, output_padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(), #2285, 20
            nn.ConvTranspose1d(20, 2, kernel_size=13, stride=7, padding=1, output_padding=1),
            #16000, 2
        )
        
    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
#%%
def norm(data):
        scale = data.max() - data.min() + 1e-8
        norm_data = (data-data.min())/scale
        return norm_data
    


# %%
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
model.train()
num_epochs = 100
loss_list = []

for i in range(num_epochs):
    losses = []
    for x, t in trainloader:
        #x = x.view(x.size(0), -1)
        x = x.to(device)
        x_hat = model(x)
        model.zero_grad()
        loss = F.mse_loss(x_hat, x)
        #loss = F.smooth_l1_loss(x, x_hat)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    loss_list.append(np.average(losses))
    print("EPOCH: {} loss: {}".format(i, np.average(losses)))

#%%
MSE = np.zeros(600)
for i, x in enumerate(testloader):
    x = x.to(device)
    x_hat = model(x)
    mse = F.mse_loss(x_hat, x)
    MSE[i] = mse
#%%
for i, x in enumerate(anomalyloader):
    x = x.to(device)
    x_hat = model(x)
    mse = F.mse_loss(x_hat, x)
    print(mse)
#%%
threshold = 0.6
## np.count_nonzero(MSE>1.0)
#%%
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss
# %%
normal_model = MyDSPNet().to(device)
trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=20)
trainer.fit(normal_model, trainloader,)
# %%
normal_model = MyDSPNet().to(device)
with open("task2_AE10.csv","w") as f:
    f.write("id,category\n")
    for i, x in enumerate(testloader):
        x = x.to(device)
        output = normal_model(x)
        pred = output.argmax(dim=1, keepdim=True)
        if MSE[i] <= threshold:
            f.write("%d,%d\n"%(i,pred.item()))
        else:
            f.write("%d,%d\n"%(i,3))