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

class Simple(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_out,options):
        super().__init__()
        self.options = options
        self.l1 = Linear(dim_in, self.options['h1'])
        self.l2 = Linear(self.options['h1'], self.options['h2'])
        self.l3 = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
                                            
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])

        self.transform_source_mask = torch.zeros(10)
        self.transform_target_mask = torch.zeros(10)
        
        self.labels = torch.zeros(10) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

        self.to_save_hidden = False
        self.hidden_dir = ''
        self.loss_add = 0

    def forward(self, x, edge_index,to_transform=False):
        # h=x
        # if self.options['dropout'] >0:
        #     h = F.dropout(h, p=self.options['dropout'], training=self.training)

        h = self.l1(x)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_l1.csv')
        h = torch.relu(h)
        if self.options['dropout'] >0:
            h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.l2(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_l2.csv')
        h = torch.relu(h)

        if to_transform: 
            if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
            h = self.transform(h)
            if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.l3(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_l3.csv')
        return h, F.softmax(h,dim=1)
    def transform(self, x):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        S_trans, self.loss_add = self.transformer.transport(S,T,
            toComputeCost=True,costType='l2')
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)
    
    def get_transport_loss(self, mask):
        loss = self.transformer.compute_loss(mask)
        return(loss)
    # def transform(self, x):
    #     # start = time.time()
    #     S = x[self.transform_source_mask]
    #     T = x[self.transform_target_mask]
    #     self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
    #     S_trans = self.transformer.transport(S,T,
    #         toComputeCost=True,costType='l2')
    #     x_trans = x.clone()
    #     if torch.isnan(S_trans).any(): 
    #         self.log.error('OT result contains NaN')
    #     else:
    #         x_trans[self.transform_source_mask] = S_trans
    #     # t = time.time()-start
    #     # self.log.info(f'transport time: {t}s ({t/60}m)')


    #     if self.to_print_transform:
    #         print_title = 'transformation'
    #         print_path = self.print_path
    #         is_print_top_target = True
    #         diff_source = self.labels[self.transform_source_mask]
    #         diff_target = self.labels[self.transform_target_mask]

    #         # printOT_Diff(S,T,S_trans,diff_source,diff_target,
    #         #     dim_red_method = 'TSNE',
    #         #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
    #         printOT_2D(S,T,S_trans,diff_source,diff_target,
    #             log=self.log,fig_path = print_path)
    #     return(x_trans)

    # def save_weights(self,dirpath):
    #     save_tensor(self.l1.weight.data, dirpath,'l1.csv')
    #     save_tensor(self.l2.weight.data, dirpath,'l2.csv')
    #     save_tensor(self.l3.weight.data, dirpath,'l3.csv')


class oldGCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_out,options):
        super().__init__()
        self.options = options
        self.gcn1 = GCNConv(dim_in, self.options['h1'])
        self.gcn2 = GCNConv(self.options['h1'], self.options['h2'])
        self.lin = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])
        self.transform_source_mask = torch.zeros(10).to(options['device'])
        self.transform_target_mask = torch.zeros(10).to(options['device'])
        self.labels = torch.zeros(10).to(options['device']) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

        self.to_save_hidden = False
        self.hidden_dir = ''
        self.loss_add = 0


    def forward(self, x, edge_index,to_transform=False,to_save_transform=False,transform_path='/'):
        h = self.gcn1(x, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gcn1.csv')
        h = torch.relu(h)
        if self.options['dropout'] >0:
            h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gcn2(h, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gcn2.csv')
        h = torch.relu(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h,to_save_transform,transform_path)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.lin(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x,to_save_transform,transform_path):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        S_trans, self.loss_add = self.transformer.transport(S,T,
            toComputeCost=True,costType='l2',toSaveTransport=to_save_transform,transportPath=transform_path)
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)
    

class GCN(torch.nn.Module):
    """Graph Convolutional Network, now with fairness option"""
    def __init__(self, dim_in, dim_out,options):
        super().__init__()
        self.options = options
        self.gcn1 = GCNConv(dim_in, self.options['h1'])
        self.gcn2 = GCNConv(self.options['h1'], self.options['h2'])
        self.lin = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])
        self.transform_source_mask = torch.zeros(10).to(options['device'])
        self.transform_target_mask = torch.zeros(10).to(options['device'])
        self.labels = torch.zeros(10).to(options['device']) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

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


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)
    
    def get_transport_loss(self, mask, fair):
        # loss = self.transformer.compute_wass_loss(mask)
        if fair: loss = self.transformer.compute_fair_sinkhorn_loss(mask)
        else: loss = self.transformer.compute_cuturi_sinkhorn_loss(mask)
        return(loss)

    # def save_weights(self,dirpath):
    #     # for param in model.parameters():
    #     #     print(param)
    #     #     print(param.data)
    #     save_tensor(self.gcn1.weight.data, dirpath,'gcn1.csv')
    #     save_tensor(self.gcn2.weight.data, dirpath,'gcn2.csv')
    #     save_tensor(self.lin.weight.data, dirpath,'lin.csv')

class GATv2(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_out, options, heads=8):
        super().__init__()
        self.options = options
        self.gat1 = GATv2Conv(dim_in, self.options['h1'], heads=heads)
        self.gat2 = GATv2Conv(self.options['h1']*heads, self.options['h2'], heads=1)
        self.lin = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])
        self.transform_source_mask = torch.zeros(10)
        self.transform_target_mask = torch.zeros(10)
        self.labels = torch.zeros(10) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

        self.to_save_hidden = False
        self.hidden_dir = ''
        self.loss_add = 0


    def forward(self, x, edge_index,to_transform=False):
        # h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.gat1(x, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gat1.csv')
        # h = F.elu(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gat2(h, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gat2.csv')
        h = torch.relu(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.lin(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        S_trans = self.transformer.transform(S,T,
            toComputeCost=True,costType='l2')
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)        


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_out, options, heads=8):
        super().__init__()
        self.options = options
        self.gat1 = GATConv(dim_in, self.options['h1'], heads=heads)
        self.gat2 = GATConv(self.options['h1']*heads, self.options['h2'], heads=1)
        self.lin = Linear(self.options['h2'], dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])
        self.transform_source_mask = torch.zeros(10)
        self.transform_target_mask = torch.zeros(10)
        self.labels = torch.zeros(10) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

        self.to_save_hidden = False
        self.hidden_dir = ''
        self.loss_add = 0

    def forward(self, x, edge_index,to_transform=False):
        # h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.gat1(x, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gat2.csv')
        # h = F.elu(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gat2(h, edge_index)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gat2.csv')
        h = torch.relu(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.lin(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        S_trans = self.transformer.transform(S,T,
            toComputeCost=True,costType='l2')
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)
    

class LinGCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_out,options):
        super().__init__()
        self.options = options
        self.gcn1 = GCNConv(dim_in, self.options['h1'],improved=False)
        self.gcn2 = GCNConv(self.options['h1'], self.options['h2'],improved=False)

        self.l1 = Linear(dim_in, self.options['h1'])
        self.l2 = Linear(self.options['h1'], self.options['h2'])

        self.final = Linear(self.options['h2']*2, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                            lr=self.options['learning_rate'],
                                            weight_decay=self.options['weight_decay'])
        self.transformer = OptimalTransport(ot_lambda=options['lambda'],
            logger = options['log'], device=options['device'])
        self.transform_source_mask = torch.zeros(10).to(options['device'])
        self.transform_target_mask = torch.zeros(10).to(options['device'])
        self.labels = torch.zeros(10).to(options['device']) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'

        self.to_save_hidden = False
        self.hidden_dir = ''
        self.loss_add = 0


    def forward(self, x, edge_index,to_transform=False):
        h_gcn = self.gcn1(x, edge_index)
        if self.to_save_hidden: save_tensor(h_gcn, self.hidden_dir,'post_gcn1.csv')
        h_gcn = torch.relu(h_gcn)
        if self.options['dropout'] >0:
            h_gcn = F.dropout(h_gcn, p=self.options['dropout'], training=self.training)
        h_gcn = self.gcn2(h_gcn, edge_index)

        h_lin = self.l1(x)
        if self.to_save_hidden: save_tensor(h_lin, self.hidden_dir,'post_lin1.csv')
        h_lin = torch.relu(h_lin)
        if self.options['dropout'] >0:
            h_lin = F.dropout(h_lin, p=self.options['dropout'], training=self.training)
        h_lin = self.l2(h_lin)

        h = torch.cat([h_gcn, h_lin], dim=1)
        
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_gcn2.csv')
        h = torch.relu(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'pre_transform.csv')
        if to_transform: h = self.transform(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_transform.csv')
        h = self.final(h)
        if self.to_save_hidden: save_tensor(h, self.hidden_dir,'post_lin.csv')
        return h, F.softmax(h,dim=1)

    def transform(self, x):
        S = x[self.transform_source_mask]
        T = x[self.transform_target_mask]
        self.log.info(f'BEGIN TRANSFORM: S shape: {S.shape}, T shape:{T.shape}')
        S_trans, self.loss_add = self.transformer.transport(S,T,
            toComputeCost=True,costType='l2')
        x_trans = x.clone()
        if torch.isnan(S_trans).any(): 
            self.log.error('OT result contains NaN')
        else:
            x_trans[self.transform_source_mask] = S_trans


        if self.to_print_transform:
            print_title = 'transformation'
            print_path = self.print_path
            is_print_top_target = True
            diff_source = self.labels[self.transform_source_mask]
            diff_target = self.labels[self.transform_target_mask]

            # printOT_Diff(S,T,S_trans,diff_source,diff_target,
            #     dim_red_method = 'TSNE',
            #     log=self.log,fig_path = print_path,title=print_title,is_top_target=is_print_top_target)
            printOT_2D(S,T,S_trans,diff_source,diff_target,
                log=self.log,fig_path = print_path)
        return(x_trans)