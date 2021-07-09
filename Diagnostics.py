import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from celluloid import Camera
from os.path import join
from SyntheticData import reverse_onehot, one_hotting
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from collections import OrderedDict


# assign one to max probability, used to get predictions
def ceiling(pred,n_labels,L=True):
    
    if not L:
        Pred=[pred]
        Max=n_labels
    else:
        Pred=pred
        Max=max(n_labels)
        
    Res=[]
    for i in range(len(Pred)):
        T=torch.zeros([len(Pred[i]),Max])
        for j in range(len(Pred[i])):
            value, indice = Pred[i][j].max(0)
            T[j, indice]=1
        Res.append(T)
    return Res


#Gives accuracy
def Score(pred,labels,n_labels,Print=False):
    value=ceiling(pred,n_labels)
    score_per_channel=[]
    IL=[]  # counts impossible labels assigned
    for i in range(len(value)):
        score=0
        il=0
        m = n_labels[i] - 1
        for j in range(len(value[i])):
            label=labels[i][j].detach().to(torch.float64)
            val=value[i][j].to(torch.float64)
            if torch.equal(val, label):
                score+=1
            else:
                if value[i][j].max(0)[1].item() > m:
                    il+=1
        score_per_channel.append(score/len(value[i]))
        IL.append(il)
    Total_score=sum(score_per_channel)/len(score_per_channel)
    
    if Print:
         print('accuracy score per channel: {}\t number of impossible labels per channel: {}\t'.format(
             score_per_channel,
             IL
             ))
         print('Total score: {}'.format(Total_score))
    return score_per_channel, Total_score

def All_Scores(model,Data,Labels,n_labels,multi=True,threshold=90,
               n_pass=100,get_all_data=False):
    
    if multi:
        Per_Value=[[0 for i in range(len(Data[0]))],[0 for i in range(len(Data[1]))]]
    else:
        Per_Value=[[0 for i in range(len(Data[0]))]]
    Get_Score=[]    
    t=0
    for k in range(n_pass):
        pred = model.forward(Data, Labels)['pyz']
        value=ceiling(pred,n_labels)
        value2=ceiling(Labels,n_labels)
        score_per_channel=[]
        for i in range(len(value)):
            score=0
            for j in range(len(value[i])):
                if torch.equal(value[i][j], value2[i][j]):
                    score+=1
                    Per_Value[i][j]+=1
            score_per_channel.append(score/len(value[i]))
        Total_score=sum(score_per_channel)/len(score_per_channel)
        Get_Score.append(score_per_channel)
        if Total_score>t:
            t=Total_score

    A=[]
    for j in range(len(Per_Value)):
        C=[]
        c=0
        for i in range(len(Per_Value[j])):
            if Per_Value[j][i]>=threshold:
                C.append(i)
                c+=1
        A.append(C)

    Var_Score=[]
    for i in range(len(Per_Value)):
        Var_Score.append(1-len(A[i])/len(Per_Value[i]))  
    
    if get_all_data:
        return Per_Value, Var_Score, A, t, Get_Score
    else:
        return Per_Value, Var_Score, A, t 


# Plot loss of individual datasets
def plot_channel_loss(Total, LL, KL, save_fig=False, out_folder=None):
    n=len(Total)
    fig=plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=1.0)
    for i in range(n):
        ax=fig.add_subplot(n, 1, i+1)
        ax.plot(Total[i], label='Total')
        ax.plot(LL[i], label='LL')
        ax.plot(KL[i], label='KL')
        plt.title('Channel n' + str(i+1) )
        plt.xlabel('number of epochs ')
        plt.ylabel('Loss')
        plt.legend()
    if save_fig:
            plt.savefig(join(out_folder,'plot_channel_loss'.format()))
    else :
        plt.show()
        

    
# Plot loss of model overall, also in relative scale. This function was ripped off from https://gitlab.inria.fr/epione_ML/mcvae
def plot_loss(model, stop_at_convergence=True, save_fig=False, out_folder=None, start=None):
    true_epochs = len(model.loss['Total_loss']) - 1
    losses = np.array([model.loss[key] for key in model.loss.keys()]).T
    if start:
        losses=losses[start:-1]
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Loss (common scale)')
    plt.xlabel('epoch')
    plt.plot(losses), plt.legend(model.loss.keys())
    if not stop_at_convergence:
        plt.xlim([0, model.epochs])
    plt.subplot(1, 2, 2)
    plt.title('loss (relative scale)')
    plt.xlabel('epoch')
    plt.plot(losses / (1e-8 + np.max(np.abs(losses), axis=0))), plt.legend(model.loss.keys())
    if not stop_at_convergence:
        plt.xlim([0, model.epochs])
    if save_fig:
        plt.savefig(join(out_folder,'loss_{}_epochs'.format(true_epochs)))
        
        
        
# Plot final latent space of the model        
def plot_latent(q, pred, labels, z_dim,  base_dim=0, save_fig=False, out_folder=None, leg=None):
    pred_lab=ceiling(pred, labels)
    lab=reverse_onehot(pred_lab)
    mark=["x","o","x","+","1","*","d","<","H","X","4","o","v","s","P"]
    for i in range(z_dim):
        plt.figure()
        for j in range(len(q)):
            plt.scatter(q[j].loc.detach().numpy()[:, base_dim], 
                        q[j].loc.detach().numpy()[:,i], 
                        c=lab[j], marker=mark[j], cmap='viridis', s=60,edgecolor='face')
        plt.xlabel('ax'+str(base_dim))
        plt.ylabel('ax'+str(i))
        plt.title('shared latent space')
        if save_fig:
            plt.savefig(join(out_folder,'plot_latent'.format()))
        else :
            plt.show()
            
            

#Generates a gif of the evolution of the latent space throughout training, saved in current directory
def plot_latent_gif(TP, L, z_dim,  base_dim=0,path='/home/jharriso/Desktop/'):
    mark=["x","+","1","*","d","<","H","X","4","o","v","s","P"]
    anim=[]
    for i in range(z_dim):
        fig = plt.figure()
        camera = Camera(fig)
        for k in range(len(TP)):
            pred_lab, q, epoch = TP[k]
            lab=reverse_onehot(pred_lab)
            for j in range(len(q)):
                plt.scatter(q[j].loc.detach().numpy()[:,  base_dim], 
                            q[j].loc.detach().numpy()[:,i], 
                            c=lab[j], marker=mark[j],cmap='viridis_r', s=25)
            camera.snap()
        animation = camera.animate(interval=100)
        anim.append(animation)
        
    for i in range(len(anim)):
        anim[i].save(path + 'celluloid_minimal_{}.gif'.format(i), writer = 'imagemagick')
        
    return anim[z_dim-1]
            
            
 # Shows the joint (or selected dataset) latent space at a specific epoch            
def plot_latent_evol(TP, L, z_dim,  base_dim=0, epoch=1000, dataset=None):
    mark=["X","o","x","+","1","*","d","<","H","X","4","o","v","s","P"]
    pred_lab, q, epoch = TP[epoch]
    lab=reverse_onehot(pred_lab)
    sns.set_style("white")
    colors = ["plum purple", "deep red", "teal", "sunflower yellow"]
    cmap = ListedColormap(sns.xkcd_palette(colors))
    for i in range(z_dim):
        if dataset:
            plt.figure()
            plt.set_facecolor('white')
            plt.scatter(q[dataset].loc.detach().numpy()[:,  base_dim], 
                        q[dataset].loc.detach().numpy()[:,i], 
                        c=lab[dataset], marker=mark[dataset],cmap=cmap, s=25)
            plt.xlabel('ax'+str(base_dim))
            plt.ylabel('ax'+str(i))
            plt.title('latent space of channel n' + str(dataset))
            
        else:
            plt.figure()
            for j in range(len(q)):
                plt.scatter(q[j].loc.detach().numpy()[:,  base_dim], 
                            q[j].loc.detach().numpy()[:,i], 
                            c=lab[j], marker=mark[j], cmap=cmap, s=25)
                plt.xlabel('ax'+str(base_dim))
                plt.ylabel('ax'+str(i))
                plt.title('shared latent space')
        plt.show()
            
        
            
# Shows the individual latent spaces 
def plot_latent_c(q, pred, labels, z_dim, base_dim=0, save_fig=False, out_folder=None):
    pred_lab=ceiling(pred, labels)
    lab=reverse_onehot(pred_lab)
    z=int(z_dim/2)
    
    if (z_dim % 2) != 0:
        zz=z+1
        
    if z_dim == 2:
        zz=z_dim
        
    else:
        zz=z
    
    for j in range(len(q)):
        fig=plt.figure()
        fig.subplots_adjust(hspace=0.5, wspace=1.0)
        for i in range(z_dim):
            ax=fig.add_subplot(zz,zz,i+1)  # (z_dim, i+1, i+1)
            ax.scatter(q[j].loc.detach().numpy()[:,base_dim], 
                       q[j].loc.detach().numpy()[:,i], 
                       c=lab[j],cmap='viridis')
        plt.title('latent space of channel n' + str(j))
        if save_fig:
            plt.savefig(join(out_folder,'plot_latent_c'.format()))
        else :
            plt.show()    
    
    
    
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
                


# Plots the data with prediction labels throughout the training   
def PlotPath(data, labels, L, n=1000, save_fig=False, out_folder=None, gif=True):
    for i in range(len(L)):
        pred_lab, q, epoch = L[i]
        if epoch % n ==0:
            PlotData(data,pred_lab, dim=0,save_fig=save_fig, out_folder=out_folder)
        if i == len(L)-1:
            PlotData(data,pred_lab, dim=0, save_fig=save_fig, out_folder=out_folder)
     
    
    
    
def GetSplit(data,label,test_size=0.1,test=False,random_state=42):
    D=data.to_numpy()
    L=label
    
    Full=torch.FloatTensor(data.values)
    Full_Label= torch.FloatTensor(label)
    
    X, X_t, Y, Y_t = train_test_split(D, L, test_size=test_size, 
                                      random_state=random_state,stratify=L)
    
    Train = torch.from_numpy(X).type(torch.FloatTensor)
    Train_label = torch.from_numpy(Y)
    Test = torch.from_numpy(X_t).type(torch.FloatTensor)
    Test_label = torch.from_numpy(Y_t)
      
    if test:
        return [Test, Test_label]
    else:
        return [Full, Full_Label, Train, Train_label, Test, Test_label]    
    
    
    
    
def KFoldSplit(data,label,split,cross=False,one_hot=True):
    D=data.to_numpy()
    L=label
    
    if cross:
        kf = StratifiedKFold(n_splits=split,shuffle=True)
        fold = kf.split(data,label)
    else:
        kf = StratifiedShuffleSplit(n_splits=split)
        fold = kf.split(data,label)
    
    S_x=[]
    S_y=[]
    S_id=[]
    for train_index, test_index in fold:
        X_train, X_test = D[train_index], D[test_index]
        y_train, y_test = L[train_index], L[test_index]
        Train = torch.from_numpy(X_train).type(torch.FloatTensor)
        Train_label = torch.from_numpy(one_hotting(y_train))
        Test = torch.from_numpy(X_test).type(torch.FloatTensor)
        Test_label = torch.from_numpy(one_hotting(y_test))

        S_x.append([Train,Test])
        S_y.append([Train_label,Test_label])
        ID_train=list(data.index[train_index])
        ID_test=list(data.index[test_index])
        S_id.append([ID_train,ID_test])
        
    return S_x, S_y,S_id


def StratifiedKFoldSplit(data,label,split):
    D=data.to_numpy()
    L=label

    kf = StratifiedKFold(n_splits=split)
    
    y=[]
    for i in L:
        if i[0]==1.:
            y.append(1)
        else:
            y.append(0)
            
    S_x=[]
    S_y=[]
    S_id=[]
    for train_index, test_index in kf.split(data,y):
        X_train, X_test = D[train_index], D[test_index]
        y_train, y_test = L[train_index], L[test_index]
        Train = torch.from_numpy(X_train).type(torch.FloatTensor)
        Train_label = torch.from_numpy(y_train)
        Test = torch.from_numpy(X_test).type(torch.FloatTensor)
        Test_label = torch.from_numpy(y_test)
        S_x.append([Train,Test])
        S_y.append([Train_label,Test_label])
        ID_train=list(data.index[train_index])
        ID_test=list(data.index[test_index])
        S_id.append([ID_train,ID_test])
        
    return S_x, S_y,S_id
    
    
    
def Save_Models(model,path,n_channels):
    SD=model.state_dict()
    for i in range(n_channels):
        State=OrderedDict()
        State['fc1.weight']=SD['fc1.'+str(i)+'.weight']
        State['fc1.bias']=SD['fc1.'+str(i)+'.bias']
        State['fc2.weight']=SD['fc2.'+str(i)+'.weight']
        State['fc2.bias']=SD['fc2.'+str(i)+'.bias']
        State['fc3.weight']=SD['fc3.'+str(i)+'.weight']
        State['fc3.bias']=SD['fc3.'+str(i)+'.bias']
        State['mu.weight']=SD['mu.'+str(i)+'.weight']
        State['mu.bias']=SD['mu.'+str(i)+'.bias']
        State['log_var.weight']=SD['log_var.'+str(i)+'.weight']
        State['log_var.bias']=SD['log_var.'+str(i)+'.bias']
        State['fc4.weight']=SD['fc4.weight']
        State['fc4.bias']=SD['fc4.bias']
        State['fc5.weight']=SD['fc5.weight']
        State['fc5.bias']=SD['fc5.bias']
        
        if type(path) is str:
            torch.save(State, path+'State_'+str(i)+'.pt')
        else:
            torch.save(State, path[i])    
    

        
    
    
    
    
    
    
    
    