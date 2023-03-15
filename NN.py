# This file includes the main steps in the DNN application:
  # split and batch data
  # hyperparameter and architecture search
  # retrain with earlystopping and logging
  # load trained models
  # visually compare outputs and labels
#%% Imports
import utilities
import numpy as np
from datetime import datetime
import keras_tuner
import math
from pathlib import WindowsPath
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
#%% Definitions
random.seed(100)
rowAmount = colAmount = 10
ds = utilities.dataManager(0, rowAmount, colAmount)
# goal = "optimalPath"
goal = "OSPlength"
#%% Data
dataTrain, dataVal = ds.TFdata(goal) # data splits
batchSize = 256
# organize both splits in batches
batchesTrain = dataTrain.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesVal = dataVal.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
# %% Hyperparameter optimization
def build_model(hp):
  inputs = keras.Input(shape = (rowAmount * colAmount,), name = "initialMap")
  # HP search defines number of hidden layers and their size
  layerSize = hp.Int("units", min_value = 32, max_value = 256, step = 64)
  numberOfLayers = hp.Int("nLayers", min_value = 2, max_value = 8, step = 2)
  x = layers.Dense(layerSize, activation = "relu")(inputs)
  for _ in range(1, numberOfLayers):
    x = layers.Dense(layerSize, activation = "relu")(x)
  if goal == "OSPlength": # if predicting length of OSP
    outputs = layers.Dense(1, name = goal)(x)
  elif goal == "optimalPath": # if predicting OSP
    outputs = layers.Dense(rowAmount * colAmount, name = goal)(x)
  model = keras.Model(inputs = inputs, outputs = outputs, name = "DNN_model")
  model.compile(
    optimizer = keras.optimizers.RMSprop(
    learning_rate = hp.Float("lr", min_value = 1e-4,
      max_value = 1, sampling = "log")
    ),
    loss = keras.losses.MeanSquaredError(),
    metrics = [keras.metrics.MeanAbsoluteError()]
  )
  return model
#%% Keras tuner API
tuner = keras_tuner.BayesianOptimization(
    build_model,
    objective = 'val_loss',
    directory = "logs\\DNN\\OSP_length" if goal == "OSPlength" else "logs\\DNN\\OptMap",
    project_name = "HPopt",
    max_trials = 20
)
# Perform HP search
tuner.search(batchesTrain, epochs = 30, validation_data = batchesVal)
# Print list of ranked HP combinations
tuner.results_summary()
#%% Train model from scratch with best hyperparameters
# Early-stop callback
earlyStopCallback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss', min_delta = 0, patience = 20, verbose = 1,
    mode = 'auto', baseline = None, restore_best_weights = True,
)
# Log folder for tensorboard
if goal == "OSPlength":
  tbPath = "logs\\DNN\\OSP_length\\" + str(len(list(WindowsPath("./logs/DNN/OSP_length").iterdir())) + 1)
elif goal == "optimalPath":
  tbPath = "logs\\DNN\\OptMap\\" + str(len(list(WindowsPath("./logs/DNN/OSP_length").iterdir())) + 1)
tensorBoardCallback = keras.callbacks.TensorBoard(
    log_dir = tbPath,
    histogram_freq = 1, write_graph = True, write_images = True,
    write_steps_per_second = True, update_freq = 'epoch'
)
# Model architecture
inputs = keras.Input(shape = (rowAmount * colAmount,), name = "initialMap")
hiddenSize = 32 if goal == "OSPlength" else 224
if goal == "OSPlength":
  x = layers.Dense(hiddenSize, activation = "relu")(inputs)
elif goal == "optimalPath":
  x = layers.Dense(hiddenSize, activation = "relu")(inputs)
hiddenL = 1 if goal == "OSPlength" else 7
for _ in range(hiddenL):
  x = layers.Dense(hiddenSize, activation = "relu")(x)
if goal == "OSPlength":
  outputs = layers.Dense(1, name = goal)(x)
elif goal == "optimalPath":
  outputs = layers.Dense(rowAmount * colAmount, name = goal)(x)
restartBestModel = keras.Model(
  inputs = inputs, outputs = outputs, name = "DNN_model"
)
restartBestModel.summary()
restartBestModel.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate = 0.0001),
    loss = keras.losses.MeanSquaredError(),
    metrics = [keras.metrics.MeanAbsoluteError()]
)
# Train model
restartBestModel.fit(
  batchesTrain, epochs = 100, callbacks = [earlyStopCallback, tensorBoardCallback],
  verbose = 1, validation_data = batchesVal
)
#%% Save model with best hyperparameters trained with early-stop
if goal == "OSPlength":
  modelPath = 'logs\\DNN\\OSP_length\\DNN_OSP_length.h5'
elif goal == "optimalPath":
  modelPath = 'logs\\DNN\\OptMap\\DNN_OSP.h5'
restartBestModel.save(modelPath)
# %% Visualize results from best models
# Load model that predicts OSP
OSPmodel = tf.keras.models.load_model('logs\\DNN\\OptMap\\DNN_OSP.h5')
# Load model that predicts OSP length
OSPlengthModel = tf.keras.models.load_model('logs\\DNN\\OSP_length\\DNN_OSP_length.h5')
ds = utilities.dataManager(0, rowAmount, colAmount)
compareData = ds.TFdata("both") # dataset with both labels
#%% Visually compare labels and outputs from trained models
for (initMapSample, OSPlengthSample, OSPsample) in compareData.take(5).batch(1):
  # use models in current sample
  map0 = np.asarray(initMapSample).reshape(-1, 1).transpose()
  predOSP = OSPmodel.predict(map0)
  predLength = OSPlengthModel.predict(map0)
  init_str = list(map(str, np.asarray(initMapSample)[0]))
  spotDict = ds.decodeMapString(init_str) # decode map definition to graphManager format
  wholeInitStr = utilities.listToStr(init_str)
  flagsRow, flagsCol = [], []
  for hexID in spotDict["flag"]:
    row = int(math.ceil((hexID + 0.1) / colAmount))
    flagsRow.append(row - 1)
    flagsCol.append(hexID - (row - 1) * colAmount)
  # instantiate graphManager
  gm = utilities.graphManager(wholeInitStr, rowAmount, colAmount,
    flagsRow, flagsCol
  )
  gm.plotMesh(0) # plot initial map
  # optimal map as list of strings
  OSPmap = gm.interpretMLout(init_str, predOSP)
  ds.decodeMapString(OSPmap) # decode map definition to graphManager format
  gm.storeMap(gm.stringToBinary(utilities.listToStr(OSPmap)))
  gm.plotMesh(1) # plot predicted optimal map
  print(f"""
  OSP length: {OSPlengthSample[0]}
  Prediction: {round(predLength[0][0])} ({
    round((predLength[0][0] / np.asarray(OSPlengthSample)[0] - 1) * 100)
  }%)
  Steps in initial map: {gm.totalSteps(gm.mapDefinition[0])}
  Steps in optimal map: {gm.totalSteps(gm.mapDefinition[1])}
  """)