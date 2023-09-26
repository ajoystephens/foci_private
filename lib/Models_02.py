import numpy as np
import torch 
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

import os

from lib.Transformer import OptimalTransport,printOT_Diff,printOT_2D

def save_tensor(h, dirpath,filename):
    if not os.path.exists(dirpath): os.makedirs(dirpath)
    filepath = dirpath+'/'+filename
    np_h = h.clone().detach().cpu().numpy()
    np.savetxt(filepath, np_h, delimiter=",")

 

class GCN(torch.nn.Module):
    """Graph Convolutional Network, now with fairness option"""
    def __init__(self, dim_in, dim_out,options,log,device):
        super().__init__()
        self.options = options
        self.gcn1 = GCNConv(dim_in, self.options['h1'])
        self.gcn2 = GCNConv(self.options['h1'], self.options['h2'])
        self.lin = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = log, device=device)
        self.transform_source_mask = torch.zeros(10).to(device)
        self.transform_target_mask = torch.zeros(10).to(device)
        self.labels = torch.zeros(10).to(device) # used only in OT to print 
        self.log = log
        self.to_print_transform = False
        self.print_path = 'test.png'
        self.device = device

        self.to_save_hidden = False
        self.hidden_dir = ''
        # self.loss_add = 0


    def forward(self, x, edge_index,to_transform=False,to_save_transform=False,transform_path='/',
                to_fair=True):
        h = self.gcn1(x, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gcn1.csv')
        h = torch.relu(h)
        if self.options['dropout'] >0:
            h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gcn2(h, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gcn2.csv')
        h = torch.relu(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h,to_save_transform,transform_path,to_fair)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.lin(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x,to_save_transform,transform_path,to_fair):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        # self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        if to_fair: 
            S_trans = self.transformer.fair_transport(S,T,
                toComputeCost=True,costType='l2',toSaveTransport=to_save_transform,transportPath=transform_path,
                fair=to_fair,gamma=self.options['gamma'],P_s=self.options['P_s'],P_t=self.options['P_t'])
        else: 
            S_trans = self.transformer.transport(S,T,
                toComputeCost=True,costType='l2',toSaveTransport=to_save_transform,transportPath=transform_path)
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans

        return(x_trans)
    
    def get_transport_loss(self, mask, fair):
        # loss = self.transformer.compute_wass_loss(mask)
        if fair: loss = self.transformer.compute_fair_sinkhorn_loss(mask)
        else: loss = self.transformer.compute_cuturi_sinkhorn_loss(mask)
        return(loss)

