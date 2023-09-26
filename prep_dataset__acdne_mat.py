import logging
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from scipy.sparse import csc_matrix


import scipy.io # to .mat files

import os
import sys
sys.path.append('lib/')
from utils import setupLogging, getArgsString

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('--dataset', default='pokec_trim_n_s', type=str)
parser.add_argument('--suffix', default='', type=str)
args = parser.parse_args()

DATASET_NAME=args.dataset
OUTPUT_FILEPATH = 'data/input_ACDNE/'+args.dataset+'.mat'
INPUT_FILEPATH = 'data/'+DATASET_NAME+'.p'

# MAT_FILEPATH = 'data/input_ACDNE/acmv9.mat' # TEST

LOGGER_NAME = 'mat_gen'
LOG_FILEPATH = 'log/data/'+LOGGER_NAME+'_'+ datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")+'.log'

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# METHODS
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# SET UP LOGGING
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
LOG = setupLogging(LOGGER_NAME,LOG_FILEPATH)
LOG.info(getArgsString(args))
LOG.info('Input File: '+INPUT_FILEPATH)
LOG.info('Output File: '+OUTPUT_FILEPATH)
# LOG.info(NOTE)



# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# INVESTIGATE MAT ---- TEST
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# LOG.info('Getting .mat data...\n\tfilepath: '+MAT_FILEPATH)

# # get source
# mat = scipy.io.loadmat(MAT_FILEPATH)

# network = mat['network']
# attrb = mat['attrb']
# group = mat['group']

# print('type, network (adj): '+str(type(network)))
# print('type, attrb (nodes): '+str(type(attrb)))
# print('type, group (labels): '+str(type(group)))


# print('shape, attrb (nodes): '+str(attrb.shape))
# print('shape, group (labels): '+str(group.shape))

# OUTPUT FROM ABOVE
# type, network (adj): <class 'scipy.sparse.csc.csc_matrix'>
# type, attrb (nodes): <class 'numpy.ndarray'>
# type, group (labels): <class 'numpy.ndarray'>

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# GET DATA
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

LOG.info('BEGIN Retreiving Data')
LOG.debug('DATASET NAME: '+DATASET_NAME)
LOG.debug('DATASET FILEPATH: '+INPUT_FILEPATH)
pickleJar = open(INPUT_FILEPATH, 'rb')     
pickles = pickle.load(pickleJar)
keys_str = 'Found the following keys in pickle file:'
for keys in pickles:
    keys_str += '\n\t'+str(keys)
LOG.debug(keys_str)
pickleJar.close()
LOG.debug(pickles['note'])
LOG.info('END Retreiving Data')

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# CONVERT DATA TYPES
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# REMINDER:
# type, network (adj): <class 'scipy.sparse.csc.csc_matrix'>
# type, attrb (nodes): <class 'numpy.ndarray'>
# type, group (labels): <class 'numpy.ndarray'>

mData = {}
mData['network'] = csc_matrix(pickles['adj'])

# nodes should already be numpy, just change name
mData['attrb'] = pickles['nodes']

# labels should already be numpy, just change name
# if pickles['labels'].shape[1] ==
print('---------->',pickles['labels'].ndim)

cnt = (pickles['labels']==0).sum()
print(f'- cnt label = 0: {cnt}')
cnt = (pickles['labels']==1).sum()
print(f'- cnt label = 1: {cnt}')

if pickles['labels'].ndim ==1:  
    mData['group'] = np.reshape(pickles['labels'],(pickles['labels'].shape[0],1))
else:
    mData['group'] = pickles['labels']

if 'has_group' in pickles.keys(): mData['has_group'] = pickles['has_group']
if 'has_label' in pickles.keys(): mData['has_label'] = pickles['has_label']

print('shape, attrb (nodes): '+str(pickles['nodes'].shape))
print('shape, group (labels): '+str(mData['group'].shape))

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# SAVE TO .MAT FILE
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
scipy.io.savemat(OUTPUT_FILEPATH, mData)
