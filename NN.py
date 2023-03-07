#%%
import utilities
from datetime import datetime
from pathlib import WindowsPath
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
#%%
random.seed(100)
rowAmount = colAmount = 10
ds = utilities.dataManager(0, rowAmount, colAmount)
goal = "OSPlength"
dataTrain, dataVal, dataTest = ds.TFdata(goal)
#%%
batchSize = 256
batchesTrain = dataTrain.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesVal = dataVal.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesTest = dataTest.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
# %%
inputs = keras.Input(shape = (rowAmount * colAmount,), name = "initialMap")
x = layers.Dense(64, activation = "relu")(inputs)
x = layers.Dense(64, activation = "relu")(x)
outputs = layers.Dense(1, name = goal)(x)
model = keras.Model(
  inputs = inputs, outputs = outputs, name = "DNN_model"
)
model.summary()
# %%
callback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0,
    patience = 0,
    verbose = 1,
    mode = 'auto',
    baseline = None,
    restore_best_weights = True,
    start_from_epoch = 0
)
tensorBoardCallback = keras.callbacks.TensorBoard(
    log_dir = str(WindowsPath("./logs/DNN")) + datetime.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq = 5,
    write_graph = True,
    write_images = True,
    write_steps_per_second = True,
    update_freq = 'epoch',
    profile_batch = 0,
    embeddings_freq = 0,
    embeddings_metadata = None
)
model.compile(
    optimizer = keras.optimizers.RMSprop(),  # Optimizer
    loss = keras.losses.MeanSquaredError(), # Loss to minimize
    # Metrics to monitor
    metrics = [keras.losses.huber(), keras.metrics.MeanAbsoluteError()]
)
#%%
history = model.fit(
  batchesTrain, epochs = 10, callbacks = [callback, tensorBoardCallback],
  verbose = 1, validation_data = batchesVal
)