#%% Imports
import os
os.environ['DGLBACKEND'] = 'pytorch'
import transfUtils
import dglUtils
import keras
import random
import data_utils
import torch.nn.functional as F
random.seed(100)
#%% Data
rowAmount = colAmount = 10 # map dimensions
ds = data_utils.dataManager(0, rowAmount, colAmount)
mapGraph = ds.DGLgraph(
    *random.choice(list(ds.TFdata("optimalPath", trainSplit = 1)[0].batch(1))),
    10, 10
)
train, val = ds.readDGLdataset(trainPercent = 0.9, batchSize = 1)
#%%
model = dglUtils.gcn(
  { # parameters
    "inDim": 1, # dimension of state embedding
    "hDim": 15, # output size of DNN in graph transformer layers
    "activFunction": F.hardswish
  }
)
#%% Instantiate graph transformer model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)