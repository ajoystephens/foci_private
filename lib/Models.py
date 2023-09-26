import torch 
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from lib.Transformer import OptimalTransport,printOT_Diff,printOT_2D


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

    def forward(self, x, edge_index,to_transform=False):
        h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.l1(h)
        h = torch.relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.l2(h)
        h = torch.relu(h)

        if to_transform: h = self.transform(h)
        h = self.l3(h)
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


class GCN(torch.nn.Module):
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
        self.transform_source_mask = torch.zeros(10)
        self.transform_target_mask = torch.zeros(10)
        self.labels = torch.zeros(10) # used only in OT to print 
        self.log = options['log']
        self.to_print_transform = False
        self.print_path = 'test.png'


    def forward(self, x, edge_index,to_transform=False):
        h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.gcn1(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        if to_transform: h = self.transform(h)
        h = self.lin(h)
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


    def forward(self, x, edge_index,to_transform=False):
        h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.gat1(x, edge_index)
        # h = F.elu(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gat2(h, edge_index)
        h = torch.relu(h)
        if to_transform: h = self.transform(h)
        h = self.lin(h)
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

    def forward(self, x, edge_index,to_transform=False):
        h = F.dropout(x, p=self.options['dropout'], training=self.training)
        h = self.gat1(x, edge_index)
        # h = F.elu(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.options['dropout'], training=self.training)
        h = self.gat2(h, edge_index)
        h = torch.relu(h)
        if to_transform: h = self.transform(h)
        h = self.lin(h)
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