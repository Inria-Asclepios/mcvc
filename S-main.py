import torch
from Modules import MCVCR
from SyntheticData import SimDataGen1, splitdata, wrong_labels, PlotData
from Diagnostics import Score, plot_channel_loss, plot_loss, plot_latent, plot_latent_c, plot_latent_evol, PlotPath


n_feats=[10,12,10,10,10]
z_dim=4

data,labels, L = SimDataGen1(n_labels=3, 
                            n_feats=n_feats,
                            n_samples=2000, 
                            class_sep=5,
                            random_labels=False, 
                            noise_size=4, 
                            weights=None)



#This line attributs wrong label to the third dataset
labl=wrong_labels(labels,dataset=2, l=2, wl=3, ratio=0.5, n=4)
X_train, X_test, Y_train, Y_test = splitdata(data,labl, 2000)
L=[4,4,4,4,4]

# Comment the three lines above and uncomment this line if you don't want the wrong label scenario
X_train, X_test, Y_train, Y_test = splitdata(data,labels, 2000)


PlotData(X_train, Y_train, dim=0)


#Set the hyper-parameters

adam_lr = 1e-4
n_epochs = 10000

init_dict = {'x_dims': n_feats, 
             'z_dim': z_dim, 
             'n_labels': L, 
             'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4 }

# Training the model

model = MCVCR(**init_dict)
model.init_loss()
model.optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
model.optimize(epochs=n_epochs, data=X_train, labels=Y_train)


# Fetches the information from the model for further analysis

pred = model.forward(X_train, Y_train)['pyz']
q=model.forward(X_train, Y_train)['qzx']
Total=model.loss_per_channel['Total_loss_per_c']
LL=model.loss_per_channel['LL_per_c']
KL=model.loss_per_channel['KL_per_c']
TP=model.Toplot

# Some plots and accuracy

Train_score = Score(pred, Y_train, L, Print=True)  #Accuracy
plot_channel_loss(Total, LL, KL)  #To check the convergence of every channels
plot_loss(model) # Check the convergence of the model overall
plot_latent_c(q,pred,L,z_dim, base_dim=0) # Shows the latent space of every dataset individually 
plot_latent(q,pred,L, z_dim, base_dim=0) # Shows the joint latent space 
PlotPath(X_train,Y_train,TP, 2000)  # Shows the data with predicted labels throughout training
plot_latent_evol(TP,L,z_dim,base_dim=0,epoch=5) # Shows the latent space at a specific time in training (epoch=)

#plot_latent_gif(TP, L, z_dim) # GIF of the evolution of the latent space, saved in directory


# Testing

pred_test = model.forward(X_test, Y_test)['pyz']
q_test=model.forward(X_test, Y_test)['qzx']
Test_score = Score(pred_test, Y_test, L, Print=True) 




# individual channels: 
    
    
X1tr=[X_train[0]]
X1te=[X_test[0]]
Y1tr=[Y_train[0]]
Y1te=[Y_test[0]]

X2tr=[X_train[1]]
X2te=[X_test[1]]
Y2tr=[Y_train[1]]
Y2te=[Y_test[1]]

X3tr=[X_train[2]]
X3te=[X_test[2]]
Y3tr=[Y_train[2]]
Y3te=[Y_test[2]]

X4tr=[X_train[3]]
X4te=[X_test[3]]
Y4tr=[Y_train[3]]
Y4te=[Y_test[3]]

X5tr=[X_train[4]]
X5te=[X_test[4]]
Y5tr=[Y_train[4]]
Y5te=[Y_test[4]]

# Channel 1:

init_dict_ch1 = {'x_dims': [n_feats[0]],
                 'z_dim': z_dim, 
                 'n_labels': [L[0]], 
                 'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4}


model_ch1 = MCVCR(**init_dict_ch1)
model_ch1.init_loss()
model_ch1.optimizer = torch.optim.Adam(model_ch1.parameters(), lr=adam_lr)
model_ch1.optimize(epochs=n_epochs, data=X1tr, labels=Y1tr)


pred_ch1 = model_ch1.forward(X1tr, Y1tr)['pyz']
q_ch1=model_ch1.forward(X1tr, Y1tr)['qzx']
Total_ch1=model_ch1.loss_per_channel['Total_loss_per_c']
LL_ch1=model_ch1.loss_per_channel['LL_per_c']
KL_ch1=model_ch1.loss_per_channel['KL_per_c']
TP_ch1=model_ch1.Toplot


Train_score_ch1 = Score(pred_ch1,Y1tr, [L[0]], Print=True) 

pred_test_ch1 = model_ch1.forward(X1te, Y1te)['pyz']
q_test_ch1=model_ch1.forward(X1te, Y1te)['qzx']

Test_score_ch1 = Score(pred_test_ch1, Y1te, [L[0]], Print=True) 


#Channel 2

init_dict_ch2 = {'x_dims': [n_feats[1]], 
                 'z_dim': z_dim, 
                 'n_labels': [L[1]], 
                 'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4}


model_ch2 = MCVCR(**init_dict_ch2)
model_ch2.init_loss()
model_ch2.optimizer = torch.optim.Adam(model_ch2.parameters(), lr=adam_lr)
model_ch2.optimize(epochs=n_epochs, data=X2tr, labels=Y2tr)


pred_ch2 = model_ch2.forward(X2tr, Y2tr)['pyz']
q_ch2=model_ch2.forward(X2tr, Y2tr)['qzx']
Total_ch2=model_ch1.loss_per_channel['Total_loss_per_c']
LL_ch2=model_ch2.loss_per_channel['LL_per_c']
KL_ch2=model_ch2.loss_per_channel['KL_per_c']
TP_ch2=model_ch2.Toplot


Train_score_ch2 = Score(pred_ch2,Y2tr, [L[1]], Print=True) 
plot_channel_loss(Total_ch2, LL_ch2, KL_ch2)
plot_loss(model_ch2)
plot_latent(q_ch2,pred_ch2, [L[1]], z_dim)

pred_test_ch2 = model_ch2.forward(X2te, Y2te)['pyz']
q_test_ch2 = model_ch2.forward(X2te, Y2te)['qzx']

Test_score_ch2 = Score(pred_test_ch2, Y2te, [L[1]], Print=True) 

# Channel 3

init_dict_ch3 = {'x_dims': [n_feats[2]], 
                 'z_dim': z_dim, 
                 'n_labels': [L[2]], 
                 'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4}

model_ch3 = MCVCR(**init_dict_ch3)
model_ch3.init_loss()
model_ch3.optimizer = torch.optim.Adam(model_ch3.parameters(), lr=adam_lr)
model_ch3.optimize(epochs=n_epochs, data=X3tr, labels=Y3tr)


pred_ch3 = model_ch3.forward(X3tr, Y3tr)['pyz']
q_ch3=model_ch3.forward(X3tr, Y3tr)['qzx']
Total_ch3=model_ch3.loss_per_channel['Total_loss_per_c']
LL_ch3=model_ch3.loss_per_channel['LL_per_c']
KL_ch3=model_ch3.loss_per_channel['KL_per_c']
TP_ch3=model_ch3.Toplot
Train_score_ch3 = Score(pred_ch3,Y3tr, [L[2]], Print=True) 
pred_test_ch3 = model_ch3.forward(X3te, Y3te)['pyz']
q_test_ch3 = model_ch3.forward(X3te, Y3te)['qzx']
Test_score_ch3 = Score(pred_test_ch3, Y3te, [L[2]], Print=True) 

# Channel 4

init_dict_ch4 = {'x_dims': [n_feats[3]], 
                 'z_dim': z_dim, 
                 'n_labels': [L[3]], 
                 'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4}


model_ch4 = MCVCR(**init_dict_ch4)
model_ch4.init_loss()
model_ch4.optimizer = torch.optim.Adam(model_ch4.parameters(), lr=adam_lr)
model_ch4.optimize(epochs=n_epochs, data=X4tr, labels=Y4tr)


pred_ch4 = model_ch4.forward(X4tr, Y4tr)['pyz']
q_ch4=model_ch4.forward(X4tr, Y4tr)['qzx']
Total_ch4=model_ch4.loss_per_channel['Total_loss_per_c']
LL_ch4=model_ch4.loss_per_channel['LL_per_c']
KL_ch4=model_ch4.loss_per_channel['KL_per_c']
TP_ch4=model_ch4.Toplot
Train_score_ch4 = Score(pred_ch4,Y4tr, [L[1]], Print=True) 
pred_test_ch4 = model_ch4.forward(X4te, Y4te)['pyz']
q_test_ch4 = model_ch4.forward(X4te, Y4te)['qzx']
Test_score_ch4 = Score(pred_test_ch4, Y4te, [L[1]], Print=True)

# Channel 5

init_dict_ch5 = {'x_dims': [n_feats[4]], 
                 'z_dim': z_dim, 
                 'n_labels': [L[4]], 
                 'h_dim1': 10, 'h_dim2': 6, 'h_dim3': 4}


model_ch5 = MCVCR(**init_dict_ch5)
model_ch5.init_loss()
model_ch5.optimizer = torch.optim.Adam(model_ch5.parameters(), lr=adam_lr)
model_ch5.optimize(epochs=n_epochs, data=X5tr, labels=Y5tr)


pred_ch5 = model_ch5.forward(X5tr, Y5tr)['pyz']
q_ch5=model_ch5.forward(X5tr, Y5tr)['qzx']
Total_ch5=model_ch5.loss_per_channel['Total_loss_per_c']
LL_ch5=model_ch5.loss_per_channel['LL_per_c']
KL_ch5=model_ch5.loss_per_channel['KL_per_c']
TP_ch5=model_ch5.Toplot
Train_score_ch5 = Score(pred_ch5,Y5tr, [L[4]], Print=True) 
pred_test_ch5 = model_ch5.forward(X5te, Y5te)['pyz']
q_test_ch5 = model_ch5.forward(X5te, Y5te)['qzx']
Test_score_ch5 = Score(pred_test_ch5, Y5te, [L[4]], Print=True) 



#Plotting every latent space on same canvas 

q_dep=[q_ch1[0],  q_ch2[0], q_ch3[0],q_ch4[0],q_ch5[0]]
pred_dep=[pred_ch1[0], pred_ch2[0],pred_ch3[0],pred_ch4[0],pred_ch5[0]]

plot_latent(q_dep, pred_dep, L, z_dim, base_dim=3)
















