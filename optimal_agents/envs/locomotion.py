from dm_control import mjcf
import numpy as np
from .base import MorphologyEnv

class VelPhysics(mjcf.Physics):
    
    def speedx(self):
        return self.named.data.sensordata['velocity'][0]

    def speedy(self):
        return self.named.data.sensordata['velocity'][1]

    def speedz(self):
        return self.named.data.sensordata['velocity'][2]

class VelocityBase(MorphologyEnv):
    PHYSICS_CLS = VelPhysics
    DEFAULT_TIME_LIMIT = 10
    
    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.action_penalty = action_penalty
        super(VelocityBase, self).__init__(morphology, **kwargs)

    def _post_step(self, action):
        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        return obs, reward, False, {}

    def _get_reward(self, obs, action):
        raise NotImplementedError

    def _reset(self, noise=True):
        # Add reset variance.
        if True:
            self._physics.data.qpos[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001
            self._physics.data.qvel[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001


class XVel(VelocityBase):
    
    def _get_reward(self, obs, action):
        reward = np.clip(self._physics.speedx(), -10, 10)
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

class NegXVel(VelocityBase):

    def _get_reward(self, obs, action):
        reward = np.clip(-1*self._physics.speedx(), -10, 10)
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

class YVel(VelocityBase):

    def _get_reward(self, obs, action):
        reward = np.clip(self._physics.speedy(), -10, 10)
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

class NegYVel(VelocityBase):

    def _get_reward(self, obs, action):
        reward = np.clip(-1*self._physics.speedy(), -10, 10)
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

class ZVel(VelocityBase):

    def _get_reward(self, obs, action):
        reward = np.clip(self._physics.speedz(), -2, 10)
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

class Directions2D(VelocityBase):
    '''
    One hot environment for 2D Locomotion agent optimization with NGE.
    '''
    
    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.cur_task = 0
        super(Directions2D, self).__init__(morphology, action_penalty=action_penalty, **kwargs)

    def _get_reward(self, obs, action):
        if self.cur_task == 0:
            reward = np.clip(self._physics.speedx(), -10, 10)
        elif self.cur_task == 1:
            reward = np.clip(-1*self._physics.speedx(), -10, 10)
        elif self.cur_task == 2:
            reward = np.clip(self._physics.speedz(), -2, 10)
        else:
            raise ValueError("Invalid Task ID")
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

    def _get_task_obs(self):
        one_hot = np.zeros(3)
        one_hot[self.cur_task] = 1
        return np.expand_dims(one_hot, axis=0)

    def _reset(self, noise=True):
        self.cur_task = (self.cur_task + 1) % 3
        super(Directions2D, self)._reset(noise=noise) # Add reset variance.

class Directions3D(VelocityBase):
    '''
    One hot environment for 3D Locomotion agent optimization with NGE.
    '''
    
    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.cur_task = 0
        super(Directions3D, self).__init__(morphology, action_penalty=action_penalty, **kwargs)

    def _get_reward(self, obs, action):
        if self.cur_task == 0:
            reward = np.clip(self._physics.speedx(), -10, 10)
        elif self.cur_task == 1:
            reward = np.clip(-1*self._physics.speedx(), -10, 10)
        elif self.cur_task == 2:
            reward = np.clip(self._physics.speedy(), -10, 10)
        elif self.cur_task == 3:
            reward = np.clip(-1*self._physics.speedy(), -10, 10)
        else:
            raise ValueError("Invalid Task ID")
        reward -= self.action_penalty * np.sum(np.square(action))
        return reward

    def _get_task_obs(self):
        one_hot = np.zeros(4)
        one_hot[self.cur_task] = 1
        return np.expand_dims(one_hot, axis=0)

    def _reset(self, noise=True):
        self.cur_task = (self.cur_task + 1) % 3
        super(Directions2D, self)._reset(noise=noise) # Add reset variance.

class LocomotionUnsupervised(MorphologyEnv):

    PHYSICS_CLS = mjcf.Physics
    DEFAULT_TIME_LIMIT = 10

    def __init__(self, morphology, global_state=False, **kwargs):
        self.global_state = global_state
        super(LocomotionUnsupervised, self).__init__(morphology, **kwargs)

    def _get_obs(self):
        # Overwritting base method _get_obs. See base.py for more details.
        # Need to overwrite to make sure that we don't get relative info and instead
        # Get the absolute state.
        if self.global_state:
            if self.use_end_sites:
                # Get the state according to end sites.
                pos = self._physics.data.xpos[self._morphology.end_site_indices[0]].copy()
            else:
                # Get the state from here.
                pos = self._physics.data.xpos[-len(self._morphology)].copy()
            obs = np.tile(np.expand_dims(pos, axis=0), (len(self._morphology), 1))
        else:
            if self.use_end_sites:
                obs = self._physics.data.xpos[self._morphology.end_site_indices].copy()
            else:
                obs = self._physics.data.xpos[-len(self._morphology):].copy()
        return dict(x=obs, edge_index=self._morphology.edge_list)

    def _post_step(self, action):
        obs = self._get_obs()
        return obs, None, False, None # Unsupervised, only return the reward.
