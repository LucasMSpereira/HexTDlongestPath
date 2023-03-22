# script used to train RL agent to change map
#%% Imports

import RLutils
from tf_agents.environments import utils

#%% 

environment = RLutils.hexEnvironment()
utils.validate_py_environment(environment, episodes=5)
# %%
