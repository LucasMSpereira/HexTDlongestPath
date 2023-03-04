#%%
import utilities
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
#%%
random.seed(100)
rowAmount = colAmount = 10
ds = utilities.dataManager(0, rowAmount, colAmount)
goal = "OSPlength"
dataTrainVal, dataTest = ds.TFdata(goal)
#%%
batchSize = 256
batchesTrainVal = dataTrainVal.batch(batch_size = batchSize,
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
tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
    **kwargs
)
model.compile(
    optimizer = keras.optimizers.RMSprop(),  # Optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(), # Loss to minimize
    metrics = [keras.metrics.SparseCategoricalAccuracy()] # Metrics to monitor
)
#%%
history = model.fit(
  batchesTrainVal, epochs = 10, callbacks = [callback],
  verbose = 1,  validation_split = 0.1
)