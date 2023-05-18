#%% Imports
import os
os.environ['DGLBACKEND'] = 'pytorch'
import transfUtils
import keras
import random
import data_utils
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
model = transfUtils.graphTransformer(
  { # parameters
    "mapGraph": mapGraph.to('cuda:0'),
    "flagAmount": 2, # number of flags in maps used in training
    "embedDim": 5, # dimension of state embedding
    "numberLayers": 5, # number of graph transformer layers
    "numberHeads": 5, # number of heads in each graph transformer layer
    "outDim": 15, # output size of DNN in graph transformer layers
    # activation function in first FNN layer of transformer layer
    "transfDNNactiv": "relu"
  }
)
#%% Instantiate graph transformer model
model.compile(
  optimizer = keras.optimizers.RMSprop(
  learning_rate = hp.Float("lr", min_value = 1e-4, max_value = 1, sampling = "log")),
  loss = keras.losses.MeanSquaredError(),
  metrics = [keras.metrics.MeanAbsoluteError()]
)