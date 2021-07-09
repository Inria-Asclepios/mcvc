from os.path import join
import numpy as np
import torch
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement



# This function a number of datasets with underlying similarities, with a lot of hyper-parameters to choose from

def SimDataGen1(n_labels=2, n_feats=[12,4], n_samples=500, class_sep=2, 
               random_labels=False, noise_size=2, weights=None):
    Data = [] 
    Labels = []
    L=[]
    n = n_feats[0]
    n_channels=len(n_feats)
    X, Y = make_classification(n_samples=n_samples, n_features=n, n_redundant= 1, 
                                 n_informative=n-1, n_clusters_per_class=1,
                                 n_classes=n_labels, class_sep=class_sep, 
                                 weights=weights)
    
    labels1 = np.zeros([n_samples, n_labels])
    for i in range(n_samples):
            labels1[i, Y[i]] = 1
    
    Data.append(torch.FloatTensor(X))
    Labels.append(torch.FloatTensor(labels1))
    L.append(n_labels)
    
    for i in range(1, n_channels):
        transform = np.random.randint(-8,8, size = n*n_feats[i]).reshape([n,n_feats[i]])
        X_trans = X.dot(transform) + noise_size*np.random.normal(size = n_samples*n_feats[i]).reshape((n_samples, n_feats[i]))
        if random_labels==True:
            l=np.random.randint(2, n_labels+1)
            lab=[]
            for j in range(len(Y)):
                if Y[j]>=l:
                    lab.append(np.random.randint(0,l))
                else: lab.append(Y[j])
            L.append(max(lab)+1)
        else: 
            lab = Y
            L=[n_labels for i in range(n_channels)]
    
        labels = np.zeros([n_samples, n_labels])
        for i in range(n_samples):
            labels[i, lab[i]] = 1

        Data.append(torch.FloatTensor(X_trans))
        Labels.append(torch.FloatTensor(labels))
        
    return Data, Labels, L


# Typical functions for processing data (not only synthetic)

def one_hotting(dx,n=None):
    if n:
        labels = np.zeros([len(dx), n])
    else:
        labels = np.zeros([len(dx), max(dx)+1])
    for i in range(len(dx)):
            labels[i, dx[i]] = 1            
    return(labels)



def reverse_onehot(Labels):
    Lab = []
    for i in range(len(Labels)):
        l = []
        for j in range(len(Labels[i])):
            for k in range(len(Labels[i][j])):
                if int(Labels[i][j][k]) == 1:
                    l.append(k)
        Lab.append(l)
    return Lab



def PlotData(Data, Labels, dim, save_fig=False, out_folder=None):
    Lab=reverse_onehot(Labels)
    for i in range(len(Data)):
            fig, ax = plt.subplots(1, Data[i].shape[1], sharex='col')
            for dim1 in range(dim , Data[i].shape[1]):
                ax[dim1].scatter(Data[i][:, dim], Data[i][:, dim1], marker='o', c=Lab[i], s=25, edgecolor='k')
                plt.xlabel('dim n' + str(dim))
                plt.ylabel('dim n' + str(dim1))
            if save_fig:
                plt.savefig(join(out_folder,'dataplot'))
            else: 
                plt.show()
        
        
        
def splitdata(data, labels, n_samples):
    n=int(n_samples/2)
    data_train=[]
    data_test=[]
    label_train=[]
    label_test=[]
    for i in range(len(data)):
        data_train.append(data[i][: n])
        data_test.append(data[i][n :])
        label_train.append(labels[i][: n])
        label_test.append(labels[i][n :])        
    return data_train, data_test, label_train, label_test


# This function was made to automate a number of scenarios of syntetic data (And run a script on clusters) no use otherwise
    
def combine(n_dims, n_channels):
    L=[]
    for i in range(len(n_channels)):
        comb=combinations_with_replacement(n_dims, n_channels[i])
        for j in list(comb): 
            L.append(j)    
    return L


# Modify one label with a new 'fake' one to be chosen.
    
def wrong_labels(labels, dataset=2, l=1, wl=2, ratio=0.5, n=None):
    LL=reverse_onehot(labels)
    for j in range(len(LL[dataset])):
                if LL[dataset][j]==l:
                    LL[dataset][j]=wl
    labl=[]
    for i in range(len(LL)):
        labl.append(torch.FloatTensor(one_hotting(LL[i],n=n)))
    return labl    



def normalize(df):
    column_maxes = df.max()
    df_max = column_maxes.max()
    return df / df_max


# Keeps the classifier of a (trained) model to be used as prior on further models
    
def meta_classifier(model, k=('fc4.weight','fc4.bias','fc5.weight','fc5.bias')):
    meta_dict=model.state_dict().fromkeys(k)
    for key in k:
        meta_dict.update({key:model.state_dict()[key]})
    return meta_dict



