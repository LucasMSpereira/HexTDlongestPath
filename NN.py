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
dataTrain, dataVal = ds.TFdata("OSPlength")
#%%
batch = data.batch(
  batch_size = 256,
  num_parallel_calls = tf.data.AUTOTUNE,
  deterministic = False
)
# %%
inputs = keras.Input(shape = (rowAmount * colAmount,))
x = layers.Dense(64, activation = "relu")(inputs)
x = layers.Dense(64, activation = "relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(
    inputs = inputs, outputs = outputs, name = "DNN_model"
)
model.summary()
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)
# %%
