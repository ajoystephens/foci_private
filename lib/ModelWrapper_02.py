import time
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import scipy.sparse as sp
from scipy.stats import linregress

from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
 
# import torch.nn.functional as F
# from torch.nn import Linear, Dropout
# from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import sys
import os
from lib.Models_02 import *
from lib.Transformer import *
from lib.utils_01 import *

# ==============================================================================
#  METHODS
# ==============================================================================

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


class ModelWrapper():
    def __init__(self,log_filepath,options,log,seed):
        self.options = options.copy()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_filepath = log_filepath
        self.fair = True
        self.log=log
        self.seed = seed

    def load_prep_data(self,source,target):
        if self.fair:
            self.log.info(f'====={" Retrieving SOURCE Data " :=<85}')
            A_s, X_s, P_s, Y_s, has_label_mask_s = load_fair_graph_data('./data/'+str(source)+'.mat',self.log)
            self.log.info(f'====={" Retrieving TARGET Data " :=<85}')
            A_t, X_t, P_t, Y_t, has_label_mask_t = load_fair_graph_data('./data/'+str(target)+'.mat',self.log)
            self.options['P_s'] = torch.tensor(P_s,dtype=torch.float).to(self.device)
            self.options['P_t'] = torch.tensor(P_t,dtype=torch.float).to(self.device)
        else: 
            self.log.info(f'====={" Retrieving SOURCE Data " :=<85}')
            A_s, X_s, Y_s, has_label_mask_s = load_graph_data('./data/'+str(source)+'.mat',self.log)
            self.log.info(f'====={" Retrieving TARGET Data " :=<85}')
            A_t, X_t, Y_t, has_label_mask_t = load_graph_data('./data/'+str(target)+'.mat',self.log)


        self.log.info(f'====={" Combine Data " :=<85}')
        N_s = A_s.shape[0]
        N_t = A_t.shape[0]
        N = N_s + N_t

        A=sp.lil_matrix((N_s+N_t,N_s+N_t),dtype=np.float32)
        A[0:N_s,0:N_s]=A_s
        A[-N_t:,-N_t:]=A_t

        X=sp.vstack((X_s, X_t))
        X = sp.lil_matrix(X).toarray()


        Y=np.concatenate((Y_s, Y_t),axis=0)
        Y_single = Y.argmax(axis=1)

        self.log.info(f'A shape: {A.shape}')
        self.log.info(f'X shape: {X.shape}')
        self.log.info(f'Y shape: {Y.shape}')

        self.log.info(f'N: {N}')
        self.log.info(f'N_s: {N_s}')
        self.log.info(f'N_t: {N_t}')

        self.num_classes = Y.shape[1]
        self.N_s = N_s
        self.N_t = N_t

        self.data_y = torch.tensor(Y_single,dtype=torch.long).to(self.device)
        self.data_x = torch.tensor(X,dtype=torch.float).to(self.device)
        self.e_ind, self.e_wei = from_scipy_sparse_matrix(A*self.options['alpha'])

        self.train_mask = np.zeros(N)
        
    def train(self,model, data):
        """Train a GNN model and return the trained model."""
        train_ce_losses = []
        train_ot_losses = []
        train_losses = []
        val_ce_losses = []
        val_ot_losses = []
        val_losses = []
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = model.optimizer
        epochs = self.options['epochs']

        # model.train()
        for epoch in range(epochs+1):
            # if((epoch % 10 == 0) and TO_SAVE_HIDDEN):
            model.train()

            to_transform = False
            if (epoch > self.options['pretrain_epochs']):
                to_transform = True
                model.transform_source_mask = data.source_mask
                model.transform_target_mask = data.target_mask
                model.labels = data.y
                model.to_print_transform = False # will print at the end but not here
            # Training
            optimizer.zero_grad()
            out, out_softmax = model(data.x, data.edge_index,to_transform=to_transform,to_save_transform=False,to_fair=self.fair)
            # loss = criterion(out[data.train_mask], data.y[data.train_mask])
            if (to_transform == True):
                ce_loss = criterion(out[data.train_mask], data.y[data.train_mask])
                ot_loss = model.get_transport_loss(self.train_mask, self.fair)
                loss = ce_loss + self.options['theta'] * ot_loss

            else: 
                ot_loss = torch.tensor(0)
                ce_loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss = ce_loss
            out_softmax = out_softmax.clone().detach().cpu().numpy()
            pred = out_softmax.argmax(axis=1)
            # pred_np = pred.clone().detach().cpu().numpy()
            pred_np = pred
            y = data.y.clone().detach().cpu().numpy()
            # print(out_softmax)
            mask = data.train_mask.clone().detach().cpu().numpy()
            acc = accuracy(pred[mask], y[mask])
            if (to_transform): start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.options['clip'])
            optimizer.step()

            # Validation
            model.eval()
            if (to_transform == True):
                val_ce_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_ot_loss = model.get_transport_loss(self.train_mask,self.fair)
                val_loss = val_ce_loss + self.options['theta'] * val_ot_loss
                # val_loss = total_loss(out[data.val_mask], data.y[data.val_mask],criterion,model.loss_add)
                # target_loss = total_loss(out[data.target_mask], data.y[data.target_mask],criterion,model.loss_add)
            else: 
                val_ot_loss = torch.tensor(0)
                val_ce_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_loss = val_ce_loss
                # val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                # target_loss = criterion(out[data.target_mask], data.y[data.target_mask])
            # val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            mask = data.val_mask.clone().detach().cpu().numpy()
            val_acc = accuracy(pred_np[mask], y[mask])

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                self.log.info(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                    f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}% | Transfer: {to_transform}')
                

            model.to_save_hidden = False


            train_ce_losses.append(ce_loss.item())
            train_ot_losses.append(ot_loss.item())
            train_losses.append(loss.item())
            val_ce_losses.append(val_ce_loss.item())
            val_ot_losses.append(val_ot_loss.item())
            val_losses.append(val_loss.item())
                # target_losses.append(target_loss.item())


        # save losses 
        dirpath = self.log_filepath+'/loss/'+str(self.seed)+'/'
        save_np(train_ce_losses, dirpath,'train_ce_losses.csv')
        save_np(train_ot_losses, dirpath,'train_ot_losses.csv')
        save_np(train_losses, dirpath,'train_losses.csv')
        save_np(val_ce_losses, dirpath,'val_ce_losses.csv')
        save_np(val_ot_losses, dirpath,'val_ot_losses.csv')
        save_np(val_losses, dirpath,'val_losses.csv')
        return model
    
    def test(self,model, data):
        """Evaluate the model on test set and print the accuracy score."""


        to_transform = True
        model.transform_source_mask = data.source_mask
        model.transform_target_mask = data.target_mask
        model.labels = data.y
        model.to_print_transform = True

        model.eval()

        out, out_softmax = model(data.x, data.edge_index,to_transform=to_transform)
        out_softmax = out_softmax.clone().detach().cpu().numpy()
        pred = out_softmax.argmax(axis=1)
        # pred_np = pred.clone().detach().cpu().numpy()
        pred_np = pred
        y = data.y.clone().detach().cpu().numpy()
        train_mask_np = data.train_mask.clone().detach().cpu().numpy()
        val_mask_np = data.val_mask.clone().detach().cpu().numpy()
        test_mask_np = data.test_mask.clone().detach().cpu().numpy()
        acc = accuracy(pred_np[test_mask_np], y[test_mask_np])
        model.to_save_hidden = False

        self.log.info(f'Train CM:\n{confusion_matrix(y[train_mask_np],pred[train_mask_np])}')
        self.log.info(f'Val CM:\n{confusion_matrix(y[val_mask_np],pred[val_mask_np])}')
        self.log.info(f'Test CM:\n{confusion_matrix(y[test_mask_np],pred[test_mask_np])}')

        dirpath = self.log_filepath+'/results/'+str(self.seed)+'/'
        save_np(y, dirpath,'y.csv')
        save_np(pred_np, dirpath,'pred.csv')
        save_np(out_softmax, dirpath,'pred_prob.csv')
        save_np(train_mask_np, dirpath,'train_mask.csv')
        save_np(val_mask_np, dirpath,'val_mask.csv')
        save_np(test_mask_np, dirpath,'test_mask.csv')

        return pred_np,y