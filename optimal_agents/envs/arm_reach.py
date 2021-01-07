from dm_control import mjcf
import numpy as np
from .base import MorphologyEnv

class ArmReach(MorphologyEnv):
    PHYSICS_CLS = mjcf.Physics
    DEFAULT_TIME_LIMIT = 4
    
    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.action_penalty = action_penalty
        self.goal_low = (0.5, -0.8)
        self.goal_high  = (1.5, 0.8)
        self.goal = np.random.uniform(low=self.goal_low, high=self.goal_high)
        super(ArmReach, self).__init__(morphology, **kwargs)
        
    def _post_step(self, action):
        obs = self._get_obs()
        cur_end_pos = self._physics.data.site_xpos[-1, :2].copy()
        dist = np.linalg.norm(cur_end_pos - self.goal)
        if dist < 0.1:
            reward = 100
            done = True
        else:
            reward = -1*dist
            reward -= self.action_penalty * np.sum(np.square(action))
            done = False
        return obs, reward, done, {'success' : float(done)}

    def _get_task_obs(self):
        return np.expand_dims(self._physics.data.xpos[-1, :2].copy() - self.goal, axis=0)

    def _reset(self, noise=True):
        # Add reset variance.
        if True:
            self._physics.data.qpos[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001
            self._physics.data.qvel[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001
        self.goal = np.random.uniform(low=self.goal_low, high=self.goal_high)
        self._physics.named.model.geom_pos['target', 'x'] = self.goal[0]
        self._physics.named.model.geom_pos['target', 'y'] = self.goal[1]

class ArmReachUnsupervised(MorphologyEnv):
    
    PHYSICS_CLS = mjcf.Physics
    DEFAULT_TIME_LIMIT = 1

    def __init__(self, morphology, global_state=False, **kwargs):
        self.global_state = global_state
        super(ArmReachUnsupervised, self).__init__(morphology, **kwargs)

    def _get_obs(self):
        if self.global_state:
            if self.use_end_sites:
                # Get the state according to end sites.
                pos = self._physics.data.xpos[self._morphology.end_site_indices[-1]].copy()
            else:
                # Get the state from here.
                pos = self._physics.data.xpos[-1].copy()
            obs = np.tile(np.expand_dims(pos, axis=0), (len(self._morphology), 1))
        else:
            if self.use_end_sites:
                obs = self._physics.data.xpos[self._morphology.end_site_indices].copy()
            else:
                obs = self._physics.data.xpos[-len(self._morphology):].copy()
        obs = dict(x=obs, edge_indx=self._morphology.edge_list)
        return obs

    def _post_step(self, action):
        return self._get_obs(), None, False, None # Unsupervised, only return the reward.
