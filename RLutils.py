# ML-related utilities

from tf_agents.environments.py_environment import PyEnvironment
import copy
import tensorflowa as tf
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from tf_agents.trajectories import time_step as ts
import numpy as np
import graph_manager
import data_utils
import utilities as utils
from tf_agents.trajectories import time_step as ts
import random

class hexEnvironment(PyEnvironment):

  """Environment for RL"""

  def __init__(self, numRow: int, numCol: int, initialMap: list, actionLimit: int):
    self.numRow = numRow # number of rows in map
    self.numCol = numCol # number of cols in map
    # action modeled as picking a value in [0, 1].
    # This interval is evenly mapped to hexagon IDs.
    # If the chosen hexagon is free (1), its state is
    # flipped (if that doesn't block the path)
    self._action_spec = BoundedArraySpec(
        shape = (), dtype = 'int32', minimum = 0,
        maximum = numRow * numCol - 1, name = 'action')
    # Current map as encoded list of integers
    self._observation_spec = BoundedArraySpec(
        shape = (numRow * numCol,), dtype = 'int32',
        minimum = 0, maximum = 5, name = 'observation')
    # shape specification for reward
    self._reward_spec = BoundedArraySpec(
        shape = (), dtype = 'int32', minimum = -1,
        maximum = 1, name = 'reward')
    self.firstMap = np.array(initialMap)
    self._state = np.array(initialMap)
    self._episode_ended = False
    self.countActions = 0
    self.actionLimit = actionLimit

  def action_spec(self):
    return self._action_spec
  
  def reward_spec(self):
    return self._reward_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    """Reset evironment"""
    self.countActions = 0
    self._episode_ended = False
    resetReturn = ts.restart(self.firstMap, reward_spec = self._reward_spec)
    print(resetReturn)
    return resetReturn

  def _step(self, action):

    """
    If the agent's action is valid, apply it to
    environment and return reward accordingly.
    """
    # spot -> unaltered state, reward = -1
    # void -> unaltered state, reward = -1
    # free
      # block -> unaltered state, reward = -1
      # acceptable
        # path improves -> altered state, reward = 1
        # path doesn't improve -> altered state, reward = -1
    # increment amount of actions taken
    self.countActions += 1
    # Make sure episodes don't go on forever
    if self.countActions > self.actionLimit:
      self._episode_ended = True
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
    print("a")
    a = ts.restart(self.firstMap, reward_spec = self._reward_spec)
    print(type(a.discount))
    print(type(a.observation))
    print(type(a.reward))
    print(type(a.step_type))
    # only non-spot free hexagons can be changed
    if self._state[action] == 1:
      # copy state
      stateCopy = list(map(str, copy.deepcopy(self._state)))
      # length of shortest path in current map
      currentMapLength = utils.pathLengthFromString(stateCopy, self.numRow, self.numCol)
      # remove hexagon chosen by agent
      stateCopy[action] = "0"
      # length of shortest path in altered map
      newMapLength = utils.pathLengthFromString(stateCopy, self.numRow, self.numCol)
      # if path wasn't blocked, but didn't improve, update state and return reward of -1
      if newMapLength == currentMapLength:
        self._state[action] = 0
        print("b")
        if self.countActions == self.actionLimit: # last action in episode
          return ts.termination(self._state, reward = np.int32(-1), discount = np.float32(1))
        return ts.transition(self._state, reward = np.int32(-1), discount = np.float32(1))
      # if path improved, update state and return reward of +1
      elif newMapLength > currentMapLength:
        self._state[action] = 0
        print("c")
        if self.countActions == self.actionLimit: # last action in episode
          return ts.termination(self._state, reward = np.int32(1), discount = np.float32(1))
        return ts.transition(self._state, reward = np.int32(1), discount = np.float32(1))
    # return reward of -1 if:
      # path was blocked
      # path didn't improve
      # hexagon chosen was a spot or already void
    print("d")
    if self.countActions == self.actionLimit: # last action in episode
      return ts.termination(self._state, reward = np.int32(-1), discount = np.float32(1))
    return ts.transition(self._state, reward = np.int32(-1), discount = np.float32(1))

def maskState(env: hexEnvironment):

    """
    Mask state, changing all IDs of spots
    and void hexagons to -1
    """

    return list(map(lambda d: -1 if d > 1 else d, copy.copy(env._state)))

class ActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActionNet, self).__init__(
        input_tensor_spec = input_tensor_spec,
        state_spec = (),
        name = 'ActionNet')
    self._output_tensor_spec = output_tensor_spec
    self._sub_layers = [
        tf.keras.layers.Dense(
            action_spec.shape.num_elements(), activation=tf.nn.tanh),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    for layer in self._sub_layers:
      output = layer(output)
    actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    return actions, network_state