# CHANGES TO THIS VERSION
# - add pre-train method
# - add alt loss

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
import pickle


from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
 
# import torch.nn.functional as F
# from torch.nn import Linear, Dropout
# from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import sys
import os
import shutil
# from lib.Models import *
from lib.Transformer import *
from lib.utils_01 import *

# from lib.FairGcnOt_02 import * 
from lib.ModelWrapper import * 

# https://towardsdatascience.com/graph-attention-networks-in-python-975736ac5c0c

###############################################################################
###############################################################################
# INITIALIZE THINGS
###############################################################################
###############################################################################
argParser = ArgumentParser()
argParser.add_argument("-s", "--source", default='pokec_trim_n_s__norm', help="source dataset")
argParser.add_argument("-t", "--target", default='pokec_trim_z_s__norm', help="target dataset")
args = argParser.parse_args()

T_SCRIPT_BEGIN = time.time()
SOURCE = args.source
TARGET = args.target

METHOD = 'GCN'
P_VAL = 0.1
TO_VALIDATE = False

FAIR = True
USE_PPMI = True


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEEDS = [1,5,10,15,20,25,30,35,40,45]
# SEEDS = [1]

NOW_STR = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")



if SOURCE == 'pokec_n' and USE_PPMI:
    OPTIONS = {
        'learning_rate': 0.001,
        'h1': 64, # dim of first hidden layer
        'h2': 32, # dim of second hidden layer
        'weight_decay': 5e-4,
        'dropout': 0.0, # check for overfitting first, if overfit then can increase
        'max_pretrain_epochs': 110,
        'max_ot_epochs': 25,
        'lambda': 0.03,
        'gamma': 100000,
        'theta': 15,
        'clip': 5 # clip gradient
    }
    PRIV_GROUP = 0 # MALE:0, FEMALE:1
    POS_LABEL = 1 # SMOKE REGULARLY: 1, DOES NOT SMOKE: 0

elif SOURCE == 'pokec_z' and USE_PPMI:
    OPTIONS = {
        'learning_rate': 0.001,
        'h1': 64, # dim of first hidden layer
        'h2': 32, # dim of second hidden layer
        'weight_decay': 5e-4,
        'dropout': 0.0, # check for overfitting first, if overfit then can increase
        'max_pretrain_epochs': 75,
        'max_ot_epochs': 25,
        'lambda': 0.03,
        'gamma': 100000, 
        'theta': 15,
        'clip': 5 # clip gradient
    }
    PRIV_GROUP = 0 # MALE:0, FEMALE:1
    POS_LABEL = 1 # SMOKE REGULARLY: 1, DOES NOT SMOKE: 0
elif SOURCE == 'compas_0' and USE_PPMI:
    OPTIONS = {
        'learning_rate': 0.001,
        'h1': 64, # dim of first hidden layer
        'h2': 32, # dim of second hidden layer
        'weight_decay': 5e-4,
        'dropout': 0.0, # check for overfitting first, if overfit then can increase
        # 'max_pretrain_epochs': 200,
        'max_pretrain_epochs': 250,
        'max_ot_epochs': 100,
        'alpha': 0.0, # Adjacency Multiplier, previous exp have found 0 best for abide and compas, 1e-3 for pokec
        # 'lambda': [0.01,0.03,0.05],
        'lambda': 0.03,
        # 'gamma': 5000,
        # 'gamma': 10000,
        'gamma': 20000, # actual
        'theta': 9,
        'clip': 5 # clip gradient
    }
    PRIV_GROUP = 0 # Caucasian:0, African American:1
    POS_LABEL = 1 # is recid: 1, not recid: 0
elif SOURCE == 'compas_1' and USE_PPMI:
    OPTIONS = {
        'learning_rate': 0.01,
        'h1': 64, # dim of first hidden layer
        'h2': 32, # dim of second hidden layer
        'weight_decay': 5e-4,
        'dropout': 0.0, # check for overfitting first, if overfit then can increase
        'max_pretrain_epochs': 60,
        'max_ot_epochs': 75,
        'alpha': 0.0, # Adjacency Multiplier, previous exp have found 0 best for abide and compas, 1e-3 for pokec
        # 'lambda': [0.01,0.03,0.05],
        'lambda': 0.03,
        # 'gamma': 5000,
        # 'gamma': 10000,
        'gamma': 20000, # actual
        'theta': 5,
        'clip': 5 # clip gradient
    }
    PRIV_GROUP = 0 # Caucasian:0, African American:1
    POS_LABEL = 1 # is recid: 1, not recid: 0
# elif SOURCE == 'abide_large' and USE_PPMI:
#     OPTIONS = {
#         'learning_rate': 0.005,
#         'h1': 64, # dim of first hidden layer
#         'h2': 32, # dim of second hidden layer
#         'weight_decay': 5e-4,
#         'dropout': 0.0, # check for overfitting first, if overfit then can increase
#         'max_pretrain_epochs': 75,
#         'max_ot_epochs': 5,
#         'alpha': 0.0, # Adjacency Multiplier, previous exp have found 0 best for abide and compas, 1e-3 for pokec
#         # 'lambda': [0.01,0.03,0.05],
#         # 'lambda': 0.05,
#         'lambda': 0.03,
#         # 'gamma': 50,
#         # 'gamma': 150,
#         'gamma': 400, # actual
#         'theta': 15,
#         'clip': 5 # clip gradient
#     }
#     PRIV_GROUP = 1 # female=0, male=1
#     POS_LABEL = 1 # is autism: 1, not autism: 0
elif SOURCE == 'credit_0' and USE_PPMI:
    OPTIONS = {
        'learning_rate': 0.001,
        'h1': 64, # dim of first hidden layer
        'h2': 32, # dim of second hidden layer
        'weight_decay': 5e-4,
        'dropout': 0.0, # check for overfitting first, if overfit then can increase
        'max_pretrain_epochs': 200,
        'max_ot_epochs': 100,
        'alpha': 1.0, # Adjacency Multiplier, previous exp have found 0 best for abide and compas, 1e-3 for pokec
        # 'lambda': [0.01,0.03,0.05],
        'lambda': 0.03,
        'gamma': 150000,
        'theta': 0.5,
        'clip': 5 # clip gradient
    }
else:
    print('Dataset not setup')
    quit()



###############################################################################
###############################################################################
# SETUP
###############################################################################
###############################################################################

# setup logging
LOGGER_NAME = 'CV'
LOG_FILEPATH = ('log/'+SOURCE+'__'+TARGET)
LOG_FILENAME = f'/main__{NOW_STR}.log'
LOG = setupLogging(LOGGER_NAME,LOG_FILEPATH,LOG_FILEPATH+LOG_FILENAME)

LOG.info(f'====={" FILE SETUP " :=<85}')
LOG.info(f'SOURCE: {SOURCE}')
LOG.info(f'TARGET: {TARGET}')
LOG.info(f'DEVICE: {DEVICE}')
LOG.info(f'====={" DONE FILE SETUP " :=<85}')

LOG.info(f'====={" LOGGING INITAL VALUES " :=<85}')
for key in OPTIONS:
    LOG.info(f'\t{key}: {OPTIONS[key]}')
LOG.info(f'====={" DONE LOGGING INITAL VALUES " :=<85}')



###############################################################################
###############################################################################
# CROSS FOLD VALIDATION
###############################################################################
###############################################################################

results={
    'train_f1':[],
    'test_f1':[],
    'train_sp':[],
    'test_sp':[],
    'train_eo':[],
    'test_eo':[],
}

seed_cnt = 1
seed_cnt_str = f'{seed_cnt:02}'

for seed in SEEDS:

    t_seed_begin = time.time()
    LOG.info(f'====={" BEGINNING SEED ("+seed_cnt_str+")" :=<85}')

    LOG.info(f'-----{" SETTING SEED " :-<85}')
    LOG.info(f'SEED: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    LOG.info(f'-----{" DONE SETTING SEED " :-<85}')

    model_wrapper = ModelWrapper(OPTIONS)
    model_wrapper.priv_group = PRIV_GROUP
    model_wrapper.pos_label = POS_LABEL

    # setup sub logger
    run_path = f'{LOG_FILEPATH}/{seed}'
    LOG.info(f'Saving details to directory: \n{run_path}')
    model_wrapper.setup_logging(run_path,'this.log')

    # other setup
    model_wrapper.set_seed(seed)

    # load data
    model_wrapper.load_data(SOURCE,TARGET,ppmi=USE_PPMI)

    # setup data
    if TO_VALIDATE: model_wrapper.set_masks_by_percent(P_VAL)
    else: model_wrapper.set_masks_full_source_target()

    # initialize
    model_wrapper.initialize_model()

    # pretrain
    LOG.info(f'Begin Pretrain')
    pre_epochs = model_wrapper.pretrain(to_save_loss=True)
    LOG.info(f'Pretrain Complete, {pre_epochs} epochs')

    # train OT
    LOG.info(f'Begin OT Train')
    ot_epochs = model_wrapper.train_ot(to_save_loss=True,to_save_hidden=True)
    LOG.info(f'OT Train Complete, {ot_epochs} epochs')

    # get test results
    pred, label = model_wrapper.test()

    # save results
    test_f1 = model_wrapper.get_f1(pred,label,mask_set='test')
    test_sp, test_eo = model_wrapper.get_fair(pred,label,mask_set='test',
                                              priv_group=PRIV_GROUP, pos_label=POS_LABEL)
    train_f1 = model_wrapper.get_f1(pred,label,mask_set='train')
    train_sp, train_eo = model_wrapper.get_fair(pred,label,mask_set='train',
                                              priv_group=PRIV_GROUP, pos_label=POS_LABEL)


    results['train_f1'].append(train_f1)
    results['test_f1'].append(test_f1)

    results['train_sp'].append(train_sp)
    results['test_sp'].append(test_sp)

    results['train_eo'].append(train_eo)
    results['test_eo'].append(test_eo)


    LOG.info(f'train f1: {train_f1}')
    LOG.info(f'test f1: {test_f1}')

    LOG.info(f'train stat par: {train_sp}')
    LOG.info(f'test stat par: {test_sp}')

    LOG.info(f'train equal opp: {train_eo}')
    LOG.info(f'test equal opp: {test_eo}')

    # wrap up
    model_wrapper.finish()

    t = time.time()-t_seed_begin
    LOG.info(f'seed time: {t}s ({t/60}m)')
    
 


LOG.info(f'====={" Summarizing Results " :=<85}')

latex_head = f'Source & Target & Method'
latex_results = f'{SOURCE} & {TARGET} & {METHOD}'

for key in results:
    mean = np.average(results[key])
    std = np.std(results[key])
    LOG.info(f'{key}: {mean:.5f} +/- {std:.5f}')

    latex_head += f' & {key}'
    latex_results += f' & {mean:.3f} \\pm {std:.3f}'

latex_head += '\\\\ \\hline'
latex_results += '\\\\ \\hline'

LOG.info('Latex Header: \n'+latex_head)
LOG.info('Latex Results: \n'+latex_results)

t = time.time()-T_SCRIPT_BEGIN
LOG.info(f'total script time: {t}s ({t/60}m) ({t/60/60}h)')
