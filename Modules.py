import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.distributions import Normal, kl
from Diagnostics import ceiling, Score
import torch.utils.data as tud

def reverse_onehot(Labels):
    Lab=[]
    for i in range(len(Labels)):
        l=[]
        for j in range(len(Labels[i])):
            for k in range(len(Labels[i][j])):
                if int(Labels[i][j][k])==1:
                    l.append(k)
        Lab.append(torch.LongTensor(l))
    return Lab


class MCVCR(nn.Module):
    def __init__(self, x_dims, z_dim, n_labels, h_dim1, h_dim2, h_dim3):
        super().__init__()

        self.x_dims = x_dims           # dimension of a data point (list)
        self.z_dim = z_dim             # dimension of latent dimensions (integer)
        self.n_labels = max(n_labels)  # maximum number of labels (list)
        self.labl = n_labels
        self.h_dim1 = h_dim1           # dim of output of first layer
        self.h_dim2 = h_dim2
        self.h_dim3= h_dim3            # dim of outpit of second layer
        self.n_channels = len(x_dims)  # number of channels (integer)

        self.init_encoder()
        self.init_classifier()
        

    def init_encoder(self):
        mu = []
        log_var = []
        fc1=[]
        fc2=[]
        fc3=[]

        for ch in range(self.n_channels):
            fc1.append(nn.Linear(self.x_dims[ch], self.h_dim1))
            fc2.append(nn.Linear(self.h_dim1, self.h_dim2))
            fc3.append(nn.Linear(self.h_dim2, self.h_dim3))
            mu.append(nn.Linear(self.h_dim3, self.z_dim))
            log_var.append(nn.Linear(self.h_dim3, self.z_dim))
        
        
        self.fc1=nn.ModuleList(fc1)
        self.fc2=nn.ModuleList(fc2)
        self.fc3=nn.ModuleList(fc3)
        self.mu = nn.ModuleList(mu)
        self.log_var = nn.ModuleList(log_var)
        

    def init_classifier(self):
        self.fc4 = nn.Linear(self.z_dim, int(self.n_labels/2))
        self.fc5 = nn.Linear(int(self.n_labels / 2), int(self.n_labels))
        

    def encoder(self, x):
        qzx = []

        for ch in range(self.n_channels):
            h = F.relu(self.fc1[ch](x[ch]))
            h = F.relu(self.fc2[ch](h))
            h = F.relu(self.fc3[ch](h))
            qzx.append(Normal(loc=self.mu[ch](h), scale=self.log_var[ch](h).exp().pow(0.5)))
        return qzx
    

    def sampling(self, qzx):
        z = []
        for ch in range(self.n_channels):
            z.append(qzx[ch].rsample())
        return z
    

    def classifier(self, z):
        pyz = []

        for ch in range(self.n_channels):
            h = F.relu(self.fc4(z[ch]))
            h = F.softmax(self.fc5(h))
            pyz.append(h)
        return pyz
    

    def forward(self, x, y):
        qzx = self.encoder(x)
        z = self.sampling(qzx)
        pyz = self.classifier(z)
        return {'x': x, 'y': y, 'qzx': qzx, 'z': z, 'pyz': pyz}
    

    def init_loss(self):
        empty_loss_1 = {
            'LL': [],
            'KL': [],
            'Total_loss': [],
        }
        # Second loss is used for the plot afterwards
        empty_loss_2 = {
            'LL_per_c': [[] for k in range(self.n_channels)],
            'KL_per_c': [[] for k in range(self.n_channels)],
            'Total_loss_per_c': [[] for k in range(self.n_channels)],
        }
        self.loss = empty_loss_1
        self.loss_per_channel = empty_loss_2
        

    def MCLoss(self, fwd_return):
        y = fwd_return['y']
        target = reverse_onehot(y)
        pyz = fwd_return['pyz']
        qzx = fwd_return['qzx']
        KL_per_c = []
        LL_per_c = []
        KL=0
        LL=0

        for i in range(self.n_channels):
            kldiv = kl.kl_divergence(qzx[i], Normal(0,1)).sum(1).mean(0)
            ll = F.cross_entropy(pyz[i], target[i], reduction='sum')
            KL+=kldiv
            LL+=ll
            KL_per_c.append(kldiv.detach().item())
            LL_per_c.append(ll.detach().item())

        T = LL + KL
        T_per_c= list(np.array(LL_per_c) - np.array(KL_per_c))
        losses = {'LL': LL,
                  'KL': KL,
                  'Total_loss': T,
                  }
        
        losses_per_channel = {'LL_per_c': LL_per_c,        # For plotting
                              'KL_per_c': KL_per_c,
                              'Total_loss_per_c': T_per_c,
                              }

        if self.training:
            self.save_loss(losses)
            self.save_channel_losses(losses_per_channel)
            return T   # Optimize over sum of channel losses

        else:
            return losses
        

    def print_loss(self, epoch):
        LLs=[self.loss_per_channel['LL_per_c'][i][-1] for i in range(len(self.loss_per_channel['LL_per_c']))]
        print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLLs: {}'.format(
            epoch,
            self.epochs[-1],
            100. * (epoch) / self.epochs[-1],
            self.loss['Total_loss'][-1],
            self.loss['LL'][-1],
            self.loss['KL'][-1],
            LLs,
        ), end='\n')
        

    def save_loss(self, losses):
        for key in self.loss.keys():
            self.loss[key].append(float(losses[key].detach().item()))
            

    def save_channel_losses(self, lpc):
        for key in self.loss_per_channel.keys():
            for i in range(self.n_channels):
                self.loss_per_channel[key][i].append(lpc[key][i])
                

    def optimize_batch(self, local_batch, local_batch_labels):

        pred = self.forward(local_batch, local_batch_labels)
        loss = self.MCLoss(pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    

    def optimize(self, epochs, data, labels, score=None, channel=None, *args, **kwargs):

        self.train()  # Inherited method which sets self.training = True

        self.epochs = [0, epochs]

        loss_value=[]
        
        self.Toplot=[]

        for epoch in range(epochs):

            loss = self.optimize_batch(data, labels)

            loss_value.append(loss)

            if np.isnan(loss):
                print('Loss is nan!')
                break

            if epoch % 100 == 0:
                self.print_loss(epoch)
                
                
            if epoch % 100 == 0:
                if score is not None:
                    s,ts=Score(self.forward(data,labels)['pyz'], labels, self.labl)
                    if channel is not None:
                        if s[channel]>score:
                            print('Objective achieved in {} \t epochs for dataset {}'.format(epoch,channel))
                            break
                    if ts>score:
                        print('Objective achieved in {} \t epochs'.format(epoch))
                        break

            if epoch % 100 == 0:
                pred = self.forward(data, labels)['pyz']
                pred_lab=ceiling(pred, self.labl)
                q= self.forward(data, labels)['qzx']
                self.Toplot.append([pred_lab, q, epoch])
                
            if epoch == epochs-1:
                pred = self.forward(data, labels)['pyz']
                pred_lab=ceiling(pred, self.labl)
                q= self.forward(data, labels)['qzx']
                self.Toplot.append([pred_lab, q, epoch])
                
                


        self.eval()  # Inherited method which sets self.training = False
        
        
        
class MCVC_light(nn.Module):
    def __init__(self, x_dims, z_dim, n_labels, h_dim1, h_dim2, h_dim3):
        super().__init__()

        self.x_dims = x_dims           # dimension of a data point (list)
        self.z_dim = z_dim             # dimension of latent dimensions (integer)
        self.n_labels =n_labels        # maximum number of labels (list)
        self.h_dim1 = h_dim1           # dim of output of first layer
        self.h_dim2 = h_dim2
        self.h_dim3= h_dim3            # dim of outpit of second layer

        self.init_encoder()
        self.init_classifier()
        

    def init_encoder(self):
        
        self.fc1=nn.Linear(self.x_dims, self.h_dim1)
        self.fc2=nn.Linear(self.h_dim1, self.h_dim2)
        self.fc3=nn.Linear(self.h_dim2, self.h_dim3)
        mu=nn.Linear(self.h_dim3, self.z_dim)
        log_var=nn.Linear(self.h_dim3, self.z_dim)

        self.mu = mu
        self.log_var = log_var
        

    def init_classifier(self):
        self.fc4 = nn.Linear(self.z_dim, int(self.n_labels/2))
        self.fc5 = nn.Linear(int(self.n_labels/2), int(self.n_labels))
        

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        qzx=Normal(loc=self.mu(h), scale=self.log_var(h).exp().pow(0.5))
        return qzx
    

    def sampling(self, qzx):
        z=qzx.rsample()
        return z
    

    def classifier(self, z):
        h = F.relu(self.fc4(z))
        h = F.softmax(self.fc5(h))
        pyz=h
        return pyz
    

    def forward(self, x, y):
        qzx = self.encoder(x)
        z = self.sampling(qzx)
        pyz = self.classifier(z)
        return pyz # {'x': x, 'y': y, 'qzx': qzx, 'z': z, 'pyz': pyz}



class Dataset(tud.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.data= data
        self.labels= labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        return X,y

class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise stop iteration 
        return tuple(out)
    
    
class MCVC_2(nn.Module):
    def __init__(self, x_dims, z_dim, n_labels, h_dim1, h_dim2, h_dim3, 
                 print_time,p=0.2): #h_dim4,
        super().__init__()

        self.x_dims = x_dims           # dimension of a data point (list)
        self.z_dim = z_dim             # dimension of latent dimensions (integer)
        self.n_labels = max(n_labels)  # maximum number of labels (list)
        self.labl = n_labels
        self.h_dim1 = h_dim1           # dim of output of first layer
        self.h_dim2 = h_dim2
        self.h_dim3 = h_dim3            # dim of outpit of second layer
        #self.h_dim4 = h_dim4
        self.n_channels = len(x_dims)  # number of channels (integer)
        self.print_time=print_time

        self.init_encoder()
        self.init_classifier()
        self.drop = nn.Dropout(p)
        self.bn = nn.BatchNorm1d(num_features=self.x_dims[0],affine=False)
        self.bn1 = nn.BatchNorm1d(num_features=self.h_dim1)
        self.bn2 = nn.BatchNorm1d(num_features=self.h_dim2)
        self.bn3 = nn.BatchNorm1d(num_features=self.h_dim3)
        

    def init_encoder(self):
        mu = []
        log_var = []
        fc1=[]
        fc2=[]
        fc3=[]
        #fc32=[]

        

        for ch in range(self.n_channels):
            fc1.append(nn.Linear(self.x_dims[ch], self.h_dim1))
            fc2.append(nn.Linear(self.h_dim1, self.h_dim2))
            fc3.append(nn.Linear(self.h_dim2, self.h_dim3))
            #fc32.append(nn.Linear(self.h_dim3, self.h_dim4))
            mu.append(nn.Linear(self.h_dim3, self.z_dim))
            log_var.append(nn.Linear(self.h_dim3, self.z_dim))
        
        
        self.fc1=nn.ModuleList(fc1)
        self.fc2=nn.ModuleList(fc2)
        self.fc3=nn.ModuleList(fc3)
        #self.fc32=nn.ModuleList(fc32)
        self.mu = nn.ModuleList(mu)
        self.log_var = nn.ModuleList(log_var)

        

    def init_classifier(self):
        self.fc4 = nn.Linear(self.z_dim, self.z_dim)
        self.fc5 = nn.Linear(self.z_dim, self.z_dim)
        self.fc6 = nn.Linear(self.z_dim, int(self.n_labels))
        

    def encoder(self, x):
        qzx = []

        for ch in range(self.n_channels):
            h=self.bn(x[ch])
            h = F.relu(self.fc1[ch](h))
            h = F.relu(self.fc2[ch](h))
            h = F.relu(self.fc3[ch](h))
            #h = F.relu(self.fc32[ch](h))
            qzx.append(Normal(loc=self.mu[ch](h), scale=self.log_var[ch](h).exp().pow(0.5)))
        return qzx
    

    def sampling(self, qzx):
        z = []
        for ch in range(self.n_channels):
            z.append(qzx[ch].rsample())
        return z
    

    def classifier(self, z):
        pyz = []
        for ch in range(self.n_channels):
            h = F.relu(self.fc4(z[ch]))
            h = F.relu(self.fc5(h))
            h = torch.sigmoid(self.fc6(h))
            pyz.append(h)
        return pyz
    

    def forward(self, x, y):
        qzx = self.encoder(x)
        z = self.sampling(qzx)
        pyz = self.classifier(z)
        return {'x': x, 'y': y, 'qzx': qzx, 'z': z, 'pyz': pyz}
    

    def init_loss(self):
        empty_loss_1 = {
            'LL': [],
            'KL': [],
            'Total_loss': [],
        }
        # Second loss is used for the plot afterwards
        empty_loss_2 = {
            'LL_per_c': [[] for k in range(self.n_channels)],
            'KL_per_c': [[] for k in range(self.n_channels)],
            'Total_loss_per_c': [[] for k in range(self.n_channels)],
        }
        self.loss = empty_loss_1
        self.loss_per_channel = empty_loss_2
        

    def MCLoss(self, fwd_return):
        y = fwd_return['y']
        target = reverse_onehot(y)
        pyz = fwd_return['pyz']
        qzx = fwd_return['qzx']
        KL_per_c = []
        LL_per_c = []
        KL=0
        LL=0

        for i in range(self.n_channels):
            kldiv = kl.kl_divergence(qzx[i], Normal(0,1)).sum(1).mean(0)
            ll = F.cross_entropy(pyz[i], target[i], reduction='sum')
            KL+=kldiv
            LL+=ll
            KL_per_c.append(kldiv.detach().item())
            LL_per_c.append(ll.detach().item())

        T = LL + KL
        T_per_c= list(np.array(LL_per_c) - np.array(KL_per_c))
        losses = {'LL': LL,
                  'KL': KL,
                  'Total_loss': T,
                  }
        
        losses_per_channel = {'LL_per_c': LL_per_c,        # For plotting
                              'KL_per_c': KL_per_c,
                              'Total_loss_per_c': T_per_c,
                              }

        if self.training:
            self.save_loss(losses)
            self.save_channel_losses(losses_per_channel)
            return T   # Optimize over sum of channel losses

        else:
            return losses
        

    def print_loss(self, epoch):
        LLs=[self.loss_per_channel['LL_per_c'][i][-1] for i in range(len(self.loss_per_channel['LL_per_c']))]
        print('====> Epoch: {:4d}/{} ({:.0f}%)\tLoss: {:.4f}\tLL: {:.4f}\tKL: {:.4f}\tLLs: {}'.format(
            epoch,
            self.epochs[-1],
            100. * (epoch) / self.epochs[-1],
            self.loss['Total_loss'][-1],
            self.loss['LL'][-1],
            self.loss['KL'][-1],
            LLs,
        ), end='\n')
        

    def save_loss(self, losses):
        for key in self.loss.keys():
            self.loss[key].append(float(losses[key].detach().item()))
            

    def save_channel_losses(self, lpc):
        for key in self.loss_per_channel.keys():
            for i in range(self.n_channels):
                self.loss_per_channel[key][i].append(lpc[key][i])
                

    def optimize_batch(self, local_batch, local_batch_labels):

        pred = self.forward(local_batch, local_batch_labels)
        loss = self.MCLoss(pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    

    def optimize(self, epochs, dataloader, score=None, channel=None, *args, **kwargs):

        self.train()  # Inherited method which sets self.training = True

        self.epochs = [0, epochs]

        loss_value=[]
        
        self.Toplot=[]

        for epoch in range(epochs):
            for i, Data in enumerate(dataloader, 0):
                y0=Data[0]
                y1=Data[1]
                data=[y0[0].float(),y1[0].float()]
                labels=[y0[1].float(),y1[1].float()]
                loss = self.optimize_batch(data,labels)

            loss_value.append(loss)

            if np.isnan(loss):
                print('Loss is nan!')
                break

            if epoch % self.print_time == 0:
                self.print_loss(epoch)
                
                
            if epoch % self.print_time == 0:
                if score is not None:
                    s,ts=Score(self.forward(data,labels)['pyz'], labels, self.labl)
                    if channel is not None:
                        if s[channel]>score:
                            print('Objective achieved in {} \t epochs for dataset {}'.format(epoch,channel))
                            break
                    if ts>score:
                        print('Objective achieved in {} \t epochs'.format(epoch))
                        break

            if epoch % self.print_time == 0:
                pred = self.forward(data, labels)['pyz']
                pred_lab=ceiling(pred, self.labl)
                q= self.forward(data, labels)['qzx']
                self.Toplot.append([pred_lab, q, epoch])
                
            if epoch == epochs-1:
                pred = self.forward(data, labels)['pyz']
                pred_lab=ceiling(pred, self.labl)
                q= self.forward(data, labels)['qzx']
                self.Toplot.append([pred_lab, q, epoch])
                
                


        self.eval()  # Inherited method which sets self.training = False