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
    def __init__(self,options):
        self.options = options.copy()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.log_filepath = log_filepath
        self.fair = True
        # self.log=log
        # self.seed = seed

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

        # self.train_mask = np.zeros(N)

    def split_data(self):
        # setup masks
        N_s = self.N_s
        N_t = self.N_t
        N = N_s+N_t

        N_train = int(N_s)
        ind_s = np.arange(N_s)
        np.random.shuffle(ind_s)
        i_train = ind_s[:N_train]
        i_val = ind_s[N_train:(N_train)]

        source_mask = np.zeros(N,dtype=bool)
        source_mask[ind_s] = True
        target_mask=np.concatenate((
            np.array(np.zeros(N_s), dtype=bool), 
            np.array(np.ones(N_t), dtype=bool)), axis=0)

        train_mask = np.zeros(N,dtype=bool)
        train_mask[i_train] = True
        val_mask = np.zeros(N,dtype=bool)
        val_mask[i_val] = True
        # train_mask=np.concatenate((
        #     np.array(np.ones(N_s), dtype=bool), 
        #     np.array(np.zeros(N_t), dtype=bool)), axis=0)
        test_mask=np.concatenate((
            np.array(np.zeros(N_s), dtype=bool), 
            np.array(np.ones(N_t), dtype=bool)), axis=0)
        
        self.train_mask = train_mask

        self.log.info(f'N_train: {N_train} (all from N_s)')
        # LOG.info(f'N_val: {N_val} (all from N_s)')

        # create torch geometric dat object
        self.data = Data(x=self.data_x,edge_index=self.e_ind,num_nodes=N,y=self.data_y)
        self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        self.data.source_mask = torch.tensor(source_mask, dtype=torch.bool)
        self.data.target_mask = torch.tensor(target_mask, dtype=torch.bool)
        self.data.validate(raise_on_error=True)
        if torch.cuda.is_available(): self.data.cuda(device=self.device)
    
    def initialize_model(self):
        self.model = GCN(self.data.num_features,self.num_classes,self.options,self.log,self.device).to(self.device)
        self.log.info(self.model)

    def setup_logging(self, log_filepath,log_filename):
        self.log = setupLogging('FGO',log_filepath,log_filepath + '/' + log_filename)
        self.results_path_root = log_filepath
        self.log_filepath = log_filepath
        
        self.log.info(f'====={" SETUP LOGGING " :=<85}')

        self.log.info(f'THIS_FILE: {os.path.basename(__file__)}')
        
        for key in self.options:
            self.log.info(f'\t{key}: {self.options[key]}')

        self.log.info(f'====={" SETUP LOGGING COMPLETE " :=<85}')


    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def pretrain(self,to_save_loss=True):
        """Train a GNN model and return the trained model."""

        train_losses = []
        val_losses = []

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.model.optimizer
        epochs = self.options['pretrain_epochs']

        for epoch in range(epochs+1):
            # if((epoch % 10 == 0) and TO_SAVE_HIDDEN):
            self.model.train()
            # Training
            optimizer.zero_grad()
            out, out_softmax = self.model(self.data.x, self.data.edge_index,
                                          to_transform=False,to_save_transform=False,to_fair=self.fair)

            ce_loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss = ce_loss

            out_softmax = out_softmax.clone().detach().cpu().numpy()
            pred = out_softmax.argmax(axis=1)

            y = self.data.y.clone().detach().cpu().numpy()

            mask = self.data.train_mask.clone().detach().cpu().numpy()
            acc = accuracy(pred[mask], y[mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options['clip'])
            optimizer.step()

            # Validation
            self.model.eval()

            val_ce_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            val_loss = val_ce_loss

            mask = self.data.val_mask.clone().detach().cpu().numpy()
            val_acc = accuracy(pred[mask], y[mask])

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                self.log.info(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                    f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}% | Transfer: {False}')
                

            self.model.to_save_hidden = False

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
                # target_losses.append(target_loss.item())


        # save losses 
        dirpath = self.log_filepath+'/loss/'+str(self.seed)+'/'
        save_np(train_losses, dirpath,'pre_train_losses.csv')
        save_np(val_losses, dirpath,'pre_val_losses.csv')
        return epoch
    
    def train_ot(self,to_save_loss=True):
        """Train a GNN model and return the trained model."""
        train_ce_losses = []
        train_ot_losses = []
        train_losses = []
        val_ce_losses = []
        val_ot_losses = []
        val_losses = []
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.model.optimizer
        epochs = self.options['ot_epochs']

        # model.train()
        for epoch in range(epochs+1):
            # if((epoch % 10 == 0) and TO_SAVE_HIDDEN):
            self.model.train()


            to_transform = True
            self.model.transform_source_mask = self.data.source_mask
            self.model.transform_target_mask = self.data.target_mask
            self.model.labels = self.data.y
            self.model.to_print_transform = False # will print at the end but not here

            # Training
            optimizer.zero_grad()
            out, out_softmax = self.model(self.data.x, self.data.edge_index,to_transform=to_transform,to_save_transform=False,to_fair=self.fair)
            # loss = criterion(out[data.train_mask], data.y[data.train_mask])
            ce_loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            ot_loss = self.model.get_transport_loss(self.train_mask, self.fair)
            loss = ce_loss + self.options['theta'] * ot_loss
            out_softmax = out_softmax.clone().detach().cpu().numpy()
            pred = out_softmax.argmax(axis=1)
            # pred_np = pred.clone().detach().cpu().numpy()
            pred_np = pred
            y = self.data.y.clone().detach().cpu().numpy()
            # print(out_softmax)
            mask = self.data.train_mask.clone().detach().cpu().numpy()
            acc = accuracy(pred[mask], y[mask])
            if (to_transform): start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options['clip'])
            optimizer.step()

            # Validation
            self.model.eval()
            val_ce_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            val_ot_loss = self.model.get_transport_loss(self.train_mask,self.fair)
            val_loss = val_ce_loss + self.options['theta'] * val_ot_loss

            mask = self.data.val_mask.clone().detach().cpu().numpy()
            val_acc = accuracy(pred_np[mask], y[mask])

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                self.log.info(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                    f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}% | Transfer: {True}')
                

            self.model.to_save_hidden = False


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
        return epochs
    def test(self):
        """Evaluate the model on test set and print the accuracy score."""


        to_transform = True
        self.model.transform_source_mask = self.data.source_mask
        self.model.transform_target_mask = self.data.target_mask
        self.model.labels = self.data.y
        self.model.to_print_transform = True

        self.model.eval()

        out, out_softmax = self.model(self.data.x, self.data.edge_index,to_transform=to_transform)
        out_softmax = out_softmax.clone().detach().cpu().numpy()
        pred = out_softmax.argmax(axis=1)
        # pred_np = pred.clone().detach().cpu().numpy()
        pred_np = pred
        y = self.data.y.clone().detach().cpu().numpy()
        train_mask_np = self.data.train_mask.clone().detach().cpu().numpy()
        val_mask_np = self.data.val_mask.clone().detach().cpu().numpy()
        test_mask_np = self.data.test_mask.clone().detach().cpu().numpy()
        acc = accuracy(pred_np[test_mask_np], y[test_mask_np])
        self.model.to_save_hidden = False

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
    
    def get_f1(self,pred_np,y,mask_set='train'):
        if mask_set == 'test':
            mask = self.data.test_mask.clone().detach().cpu().numpy()
        elif mask_set == 'val':
            mask = self.data.val_mask.clone().detach().cpu().numpy()
        else: #default to training
            mask = self.data.train_mask.clone().detach().cpu().numpy()
        
        micro_f1 = f1_score(y[mask], pred_np[mask], average='micro')
        self.log.info(f'{mask_set} Micro F1: {micro_f1}')
        macro_f1 = f1_score(y[mask], pred_np[mask], average='macro')
        self.log.info(f'{mask_set} Macro F1: {macro_f1}')

        return macro_f1, micro_f1
    
    def get_fair(self,pred_np,y,mask_set='train',priv_group=1, pos_label=1):
        if mask_set == 'test':
            mask = self.data.test_mask.clone().detach().cpu().numpy()
        elif mask_set == 'val':
            mask = self.data.val_mask.clone().detach().cpu().numpy()
        else: #default to training
            mask = self.data.train_mask.clone().detach().cpu().numpy()
        
        P_s = self.options['P_s'].clone().detach().cpu().numpy()
        P_t = self.options['P_t'].clone().detach().cpu().numpy()
        P=np.concatenate((P_s, P_t),axis=0)
        
        sp = getStatParity(y[mask], pred_np[mask], P[mask],
                           priv_group=priv_group, pos_label=pos_label)
        self.log.info(f'{mask_set} Statistical Parity: {sp}')

        eo = getEqualOpportunity(y[mask], pred_np[mask], P[mask],
                           priv_group=priv_group, pos_label=pos_label)
        self.log.info(f'{mask_set} Equal Opportunity: {eo}')

        return sp,eo

    def finish(self):
        # self.log.close()
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)