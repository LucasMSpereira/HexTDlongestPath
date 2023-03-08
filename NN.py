#%%
import utilities
from datetime import datetime
import keras_tuner
from pathlib import WindowsPath
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
#%%
random.seed(100)
rowAmount = colAmount = 10
ds = utilities.dataManager(0, rowAmount, colAmount)
goal = "optimalPath"
dataTrain, dataVal = ds.TFdata(goal)
batchSize = 256
batchesTrain = dataTrain.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesVal = dataVal.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
# %%
def build_model(hp):
  inputs = keras.Input(shape = (rowAmount * colAmount,), name = "initialMap")
  # HP search defines size of hidden layers
  layerSize = hp.Int("units", min_value = 32, max_value = 256, step = 64)
  numberOfLayers = hp.Int("nLayers", min_value = 2, max_value = 8, step = 2)
  x = layers.Dense(layerSize, activation = "relu")(inputs)
  for _ in range(1, numberOfLayers):
    x = layers.Dense(layerSize, activation = "relu")(x)
  # outputs = layers.Dense(1, name = goal)(x)
  outputs = layers.Dense(rowAmount * colAmount, name = goal)(x)
  model = keras.Model(
    inputs = inputs, outputs = outputs, name = "DNN_model"
  )
  model.compile(
    optimizer = keras.optimizers.RMSprop(
    learning_rate = hp.Float("lr", min_value = 1e-4,
      max_value = 1, sampling = "log")
    ),
    loss = keras.losses.MeanSquaredError(),
    metrics = [keras.metrics.MeanAbsoluteError()]
  )
  return model
#%%
tuner = keras_tuner.BayesianOptimization(
    build_model,
    objective = 'val_loss',
    # directory = "logs\\DNN\\OSP_length",
    directory = "logs\\DNN\\OptMap",
    project_name = "HPopt",
    max_trials = 20
)
tuner.search(batchesTrain, epochs = 30, validation_data = batchesVal)
tuner.results_summary()
#%%
earlyStopCallback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss', min_delta = 0, patience = 20, verbose = 1,
    mode = 'auto', baseline = None, restore_best_weights = True,
)
tensorBoardCallback = keras.callbacks.TensorBoard(
    log_dir = "logs\\DNN\\OptMap\\" +
      str(len(list(WindowsPath("./logs/DNN/OptMap").iterdir())) + 1),
    histogram_freq = 1, write_graph = True, write_images = True,
    write_steps_per_second = True, update_freq = 'epoch'
)
inputs = keras.Input(shape = (rowAmount * colAmount,), name = "initialMap")
x = layers.Dense(224, activation = "relu")(inputs)
for _ in range(1):
  x = layers.Dense(224, activation = "relu")(x)
# outputs = layers.Dense(1, name = goal)(x)
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
restartBestModel.fit(
  batchesTrain, epochs = 100, callbacks = [earlyStopCallback, tensorBoardCallback],
  verbose = 1, validation_data = batchesVal
)
# %%
