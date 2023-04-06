# script used to train RL agent to change map
#%% Imports
import RLutils
import tf_agents
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import sequential
import numpy as np
import data_utils
from tf_agents.agents.dqn import dqn_agent
import random
#%% Initializations and data
random.seed(100)
episodeAmount = 1
stepsPerEpisode = 10
rowAmount = colAmount = 10
ds = data_utils.dataManager(0, rowAmount, colAmount)
# samples are (encoded map definition as integer list, OSP length) pairs
mapData, _ = ds.TFdata("OSPlength")
# shuffle and batch data
mapData = list(mapData.shuffle(len(mapData)).batch(1))
# Test environment
hexEnv = RLutils.hexEnvironment(
  rowAmount, colAmount, list(np.asarray(list(mapData[0][0][0]))), 10
)
# utils.validate_py_environment(hexEnv, episodes = 5)
hexEnv = TFPyEnvironment(hexEnv)
#%%
# disc = np.float32(1.0)
# discSpec = tf_agents.specs.BoundedArraySpec(
#   shape = (), dtype = 'float32', name='discount',
#   minimum=0.0, maximum=1.0
# )
# print(f'discSpec {discSpec.check_array(disc)}')
# obs = np.random.randint(0, 6, 100)
# obsSpec = tf_agents.specs.BoundedArraySpec(
#         shape = (100,), dtype = np.int32,
#         minimum = 0, maximum = 5, name = 'observation')
# print(f'obsSpec {obsSpec.check_array(obs)}')
# reward = np.int32(-1)
# rewardSpec = tf_agents.specs.BoundedArraySpec(
#         shape = (), dtype = 'int32', minimum = -1,
#         maximum = 1, name = 'reward')
# print(f'rewardSpec {rewardSpec.check_array(reward)}')
# step = np.int32(-1)
# stepSpec = ts.time_step_spec(obsSpec, rewardSpec)
# print(f'stepSpec {stepSpec.check_array(step)}')
#%% Agent
q_net = sequential.Sequential(
  [
    tf.keras.layers.Dense(
      num_units,
      activation = tf.keras.activations.relu,
      kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale = 2.0, mode = 'fan_in', distribution = 'truncated_normal'
      )
    ) for num_units in (100, 50)
  ] +
  [
    tf.keras.layers.Dense(
      num_actions,
      activation = None,
      kernel_initializer = tf.keras.initializers.RandomUniform(
        minval = -0.03, maxval = 0.03
      ),
      bias_initializer = tf.keras.initializers.Constant(-0.2)
    )
  ]
)
agent = dqn_agent.DqnAgent(
    hexEnv.time_step_spec(),
    hexEnv.action_spec(),
    q_network = q_net,
    optimizer = optimizer,
    td_errors_loss_fn = common.element_wise_squared_loss,
    train_step_counter = train_step_counter)
#%% Agent interacts with environment
for ep in range(hexEnv.actionLimit):
  action = dqn_agent()

  
get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0
for i in range(num_steps):
  action = tf.constant([i % 2])
  # applies the action and returns the new TimeStep.
  next_time_step = tf_env.step(action)
  transitions.append([time_step, action, next_time_step])
  reward += next_time_step.reward
  time_step = next_time_step

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
print('\n'.join(map(str, np_transitions)))
print('Total reward:', reward.numpy())