# IRMAE - WD

This repo contains a demo for Implicit Rank Minimizing Autoencoder with Weight-Decay (IRMAE-WD) implemented in pytorch applied to the KSE, L=22 dataset. This repo accompanies the work outlined in the following paper: 	arXiv:2305.01090

-----------------------------------------------

**IRMAEWD_AE.pt**  

Pytorch model trained for 200 epochs

-----------------------------------------------

**IRMAE_WD.py**

Pytorch implementation of IRMAE-WD applied to KSE, L=22 dataset

-----------------------------------------------

**KSE_L22.mat**

KSE, L=22 dataset

-----------------------------------------------

**Out.txt**

Output of training log

-----------------------------------------------

**TrainCurve.png**

Training log

-----------------------------------------------

**code_musigma.p**

Mean and standard deviation of learned code space

-----------------------------------------------

**code_sValues.png**

Singular value spectra of the covariance of latent space z (this is a demo for 200 epochs, increase epochs to 1000 for sharper drop)

-----------------------------------------------

**code_svd.p**

SVD matrices U, S, V of the covariance of the learned latent space

-----------------------------------------------

**err.p**

Training log data

-----------------------------------------------

**training_svd.p**
Singular value spectra of the covariance of z as a function of training epochs

-----------------------------------------------
