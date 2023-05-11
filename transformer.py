#%% Imports
import os
os.environ['DGLBACKEND'] = 'pytorch'
import transfUtils
import keras
import dgl
import random
import tensorflow as tf
import math
import data_utils
import torch
random.seed(100)
#%%
ds = data_utils.dataManager(0, 10, 10)
train, val = ds.readDGLdataset(trainPercent = 0.8, batchSize = 64)
#%% Data
rowAmount = colAmount = 10 # map dimensions
ds = data_utils.dataManager(0, rowAmount, colAmount)
dataTrain, dataVal = ds.readDGLdataset(0.8) # dataset
batchSize = 100
# organize both splits in batches
batchesTrain = dataTrain.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesVal = dataVal.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
#%%
model = transfUtils.graphTransformer(
  { # parameters
    "map": 1,
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