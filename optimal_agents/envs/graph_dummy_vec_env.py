from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import numpy as np

from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

def graph_obs_space(space):
    boxes = {}
    for key in space.spaces.keys():
        if key == 'edge_index':
            boxes[key] = spaces.Box(low=0, high=1000, shape=(1,2), dtype=np.long)
        else:
            boxes[key] = spaces.Box(low=space[key].low[0:1], high=space[key].high[0:1], dtype=space[key].dtype)
    space = spaces.Dict(**boxes)
    return space

def graph_action_space():
    return spaces.Box(low=-1, high=1, shape=(1,))

class GraphDummyVecEnv(VecEnv):
    """
    Creates a vectorized wrapper for multiple environments for graph observations
    Based on: https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/vec_env/dummy_vec_env.py
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        boxes = {}
        # Make observation space 1 x d, representing a single node.
        action_space = graph_action_space()
        observation_space = graph_obs_space(env.observation_space)
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs_list = [None for _ in range(self.num_envs)] # Allocate at start for SPEED
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            # self._save_obs(env_idx, obs) REMOVE
            obs_list[env_idx] = obs
        return (obs_list, np.copy(self.buf_rews), np.copy(self.buf_dones),
                deepcopy(self.buf_infos))

    # def get_morphology_obs(self, morphology):
    #     return [env.get_morphology_obs(morphology, include_segments=False) for env in self.envs]

    def seed(self, seed=None):
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        obs_list = [None for _ in range(self.num_envs)] # Allocate at start for SPEED
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            # self._save_obs(env_idx, obs)
            obs_list[env_idx] = obs
        return obs_list

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode: str = 'human'):
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]