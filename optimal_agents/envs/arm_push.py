from dm_control import mjcf
import numpy as np
from .base import MorphologyEnv

class ArmPushBase(MorphologyEnv):
    PHYSICS_CLS = mjcf.Physics
    DEFAULT_TIME_LIMIT = 4

    GOAL = None

    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.action_penalty = action_penalty
        super(ArmPushBase, self).__init__(morphology, **kwargs)

    def _post_step(self, action):
        obs = self._get_obs()
        cur_box_pos = self._get_box_position()
        dist = np.linalg.norm(cur_box_pos - self.GOAL)
        arm_dist = np.linalg.norm(cur_box_pos - self._physics.data.site_xpos[-1, :2].copy())
        if dist < 0.05:
            reward = 100
            done = True
        else:
            reward = -1 * dist
            reward -= 0.5*arm_dist
            reward -= self.action_penalty * np.sum(np.square(action))
            done = False

        return obs, reward, done, {}

    def _get_task_obs(self):
        return np.expand_dims(np.concatenate((
            self._get_box_position() - self.GOAL,
            self._get_box_position(),
            self._get_box_position() - self._physics.data.site_xpos[-1, :2].copy()
        )), axis=0)

    def _reset(self, noise=True):
        # Add reset variance.
        if True:
            self._physics.data.qpos[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001
            self._physics.data.qvel[-self._morphology.num_joints:] = self.np_random.randn(self._morphology.num_joints) * 0.001

    def _get_box_position(self):
        raise NotImplementedError

class ArmPush1G1(ArmPushBase):
    GOAL = np.array([0.0, 0.65])

    def _get_box_position(self):
        return self._physics.data.xpos[1, :2].copy()

class ArmPush1G2(ArmPushBase):
    GOAL = np.array([0.9, 1.4])

    def _get_box_position(self):
        return self._physics.data.xpos[1, :2].copy()

class ArmPush1G3(ArmPushBase):
    GOAL = np.array([0.15, 1.25])

    def _get_box_position(self):
        return self._physics.data.xpos[1, :2].copy()

class ArmPush2G1(ArmPushBase):
    GOAL = np.array([0.0, -0.65])

    def _get_box_position(self):
        return self._physics.data.xpos[2, :2].copy()

class ArmPush2G2(ArmPushBase):
    GOAL = np.array([0.9, -1.4])

    def _get_box_position(self):
        return self._physics.data.xpos[2, :2].copy()

class ArmPush2G3(ArmPushBase):
    GOAL = np.array([0.15, -1.25])

    def _get_box_position(self):
        return self._physics.data.xpos[2, :2].copy()

class ArmPushAll(ArmPushBase):

    GOALS = np.array([
                        [0.0, 0.65],
                        [0.9, 1.4],
                        [0.15, 1.25],
                        [0.0, -0.65],
                        [0.9, -1.4],
                        [0.15, -1.25]
                    ])

    def __init__(self, morphology, action_penalty=0.0, **kwargs):
        self.cur_task = 0
        self.GOAL = GOALS[self.cur_task]
        super(ArmPushBase, self).__init__(morphology, action_penalty=action_penalty, **kwargs)

    def _get_box_position(self):
        if self.cur_task < 3:
            return self._physics.data.xpos[1, :2].copy()
        else:
            return self._physics.data.xpos[2, :2].copy()

    def _reset(self, noise=True):
        self.cur_task = (self.cur_task + 1) % 3
        self.GOAL = self.GOALS[self.cur_task]
        super(ArmPushAll, self)._reset(noise=noise) # Add reset variance.

class ArmPushUnsupervised(MorphologyEnv):
    
    PHYSICS_CLS = mjcf.Physics
    DEFAULT_TIME_LIMIT = 1.25

    def __init__(self, morphology, **kwargs):
        super(ArmPushUnsupervised, self).__init__(morphology, **kwargs)

    def _get_obs(self):
        x_obs = np.concatenate((
                    self._physics.data.xpos[1].copy(),
                    self._physics.data.xpos[2].copy(),
                ), axis=0) # Get the box positions
        x_obs = np.tile(np.expand_dims(x_obs, axis=0), (len(self._morphology), 1))
        obs = dict(x=x_obs, edge_index=self._morphology.edge_list)
        return obs

    def _post_step(self, action):
        return self._get_obs(), None, False, None # Unsupervised, only return the reward.
