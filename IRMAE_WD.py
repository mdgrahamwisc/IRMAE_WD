#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Kevin Zeng

###############################################################################
                        IRMAE-WD Code
###############################################################################

This code considers data from the Kuramoto-Sivashinsky equation (KSE), L=22. The 
number of linear layers can be changed in the autoencoder portion and the 
weight decay can be changed in the variable wd. 


###############################################################################

"""



import numpy as np
import matplotlib.pyplot as plt
import pickle as p
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch as T
from torch.utils.data import DataLoader

class autoencoder(nn.Module):
    def __init__(self, ambient_dim=64, code_dim=20, filepath='testae'):
        super(autoencoder, self).__init__()
        
        self.ambient_dim = ambient_dim
        self.code_dim = code_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.ambient_dim, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,self.code_dim),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False))
        
        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.ambient_dim))
        
    def forward(self,x):
        code = self.encoder(x)
        xhat = self.decoder(code)
        return xhat
    
    def encode(self, x):
        code = self.encoder(x)
        return code
    
    def decode(self, code):
        xhat = self.decoder(code)
        return xhat

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    

device = T.device("cuda" if T.cuda.is_available() else "cpu")

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

# Get Covariance, SVD
def getSVD(code_data):
    #Compute covariance matrix and singular values
    code_mean = code_data.mean(axis=0)
    code_std = code_data.std(axis=0)
    code_data = (code_data - code_mean)
    
    covMatrix = (code_data.T @ code_data) / len(dataset)
    u, s, v = np.linalg.svd(covMatrix, full_matrices=True)

    return code_mean, code_std, covMatrix, u, s, v


if __name__ == '__main__':
    
    #Parameters
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-3
    train = True

    #Weight decay parameter
    wd = 6
    wd_param = 10**-wd
    

    #Initialize model, define loss function, set optimizer
    model = autoencoder().to(device)
    model.double()
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd_param)

    #Load data for Kuramoto-Sivashinsky equation (KSE), L=22
    matdata = scipy.io.loadmat('./KSE_L22.mat')

    rawdata = matdata['ut']
    rawdata = rawdata[0:64,100:40000]
    rawdata = rawdata.T
    mean = rawdata.mean(axis=0)
    std = rawdata.std(axis=0)

    #Normalize Data with mean and std
    dataset = (rawdata - mean)/std
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Data for Computing SVD
    chunk = np.copy(dataset)
    Testdata = T.tensor(chunk, dtype=T.double).to(device)

    #Initialize Storage Matrices
    tot_err = []
    s_save = np.array([])

    if train:
        for epoch in range(num_epochs):
            for snapshots in dataloader:
                inputs = snapshots
                inputs = inputs.to(device)
                
                #Forward pass, compute loss
                reconstructed = model(inputs)
                loss = loss_function(reconstructed, inputs)
                
                #Back propagation step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.6f}'
                .format(epoch + 1, num_epochs, loss.item()))
            tot_err.append(loss.item())

            # Print out the Loss and the time the computation took
            if epoch % 10 == 0:
                output('Iter {:04d} | Total Loss {:.6f} '.format(epoch, loss.item())+'\n')
                code_data = model.encode(Testdata)
                code_data = code_data.detach().numpy()
                _, _, temp_cov, _, temp_s, _ = getSVD(code_data)
                s_save = np.hstack([s_save, temp_s[:,np.newaxis]]) if s_save.size else temp_s[:,np.newaxis]


        T.save(model.state_dict(), 'IRMAE_AE.pt')
        p.dump(tot_err,open('err.p','wb'))
                #Print Training Curve
        fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
        plt.semilogy(tot_err,c='k', label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('TrainCurve.png')
        
    else:
        model.load_state_dict(T.load('IRMAE_AE.pt'))
        print('Testing')
        
    #Get data for plotting and computing singular values/basis vectors
    code_data = model.encode(Testdata)
    code_data = code_data.detach().numpy()
    code_mean, code_std, covMatrix, u, s, v = getSVD(code_data)

    #Save Results
    p.dump([code_mean,code_std],open('code_musigma.p','wb'))
    p.dump([u,s,v],open('code_svd.p','wb'))
    p.dump(s_save,open('training_svd.p','wb'))
    
    #Plotting singular values
    fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
    plt.semilogy(s/s[0],'ko--')
    plt.xlabel('Singular value')
    plt.ylabel('Singular value of covariance of z')
    plt.tight_layout()
    plt.savefig(open('code_sValues.png','wb'))

