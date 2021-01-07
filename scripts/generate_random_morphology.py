import numpy as np
from dm_control import viewer

from optimal_agents.morphology import Morphology
from optimal_agents.morphology import random2d
from optimal_agents.envs.dm_control_env import dm_control_test_env
from optimal_agents.morphology import arenas

global_kwargs = {"option.timestep": 0.01}
geom_kwargs = {"contype": 1,
                   "conaffinity": 1,
                   "condim": 3,
                   "friction": [0.4, 0.1, 0.1],
                   }

joint_kwargs = {"damping" : 2,
                    "armature" : 0.1,
                    "stiffness": 20}

morphology = random2d(mutation_kwargs={})

env = dm_control_test_env(morphology, arena=arenas.GM_Terrain())

action_spec = env.action_spec()
def random_policy(time_step):
    del time_step  # Unused.
    return np.random.uniform(low=action_spec.minimum,
                                high=action_spec.maximum,
                                size=action_spec.shape)

viewer.launch(env, policy=random_policy)
