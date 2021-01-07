import numpy as np
import gym
from collections import OrderedDict
from optimal_agents.morphology import Morphology, Node
from dm_control.rl.control import compute_n_steps

class MorphologyEnv(gym.Env):
    '''
    Base class for morphology environment. Accomplishes the following:
    1. Implements Open AI Gym interface though underlying simulation done with DM Control
    2. Allows for dynamically changing morphology each episode of training. 
        This means dynamic updating of the action space and observation space.
    '''

    PHYSICS_CLS = None
    DEFAULT_TIME_LIMIT = None

    def __init__(self, morphology, arena=None, pad_actions=False, allow_exceptions=False, use_end_sites=False,
                       time_limit=None, control_timestep=None, n_sub_steps=None):
        self._morphology = morphology
        self.use_end_sites = use_end_sites
        self._arena = arena
        self.pad_actions = pad_actions
        self.allow_exceptions = allow_exceptions
        assert not self.PHYSICS_CLS is None, "Did not supply a DM Control Physics class for simulation"
        mjcf_model = self._morphology.construct(arena=self._arena)
        self._physics = self.PHYSICS_CLS.from_mjcf_model(mjcf_model)

        if n_sub_steps is not None and control_timestep is not None:
            raise ValueError('Both n_sub_steps and control_timestep were supplied.')
        elif n_sub_steps is not None:
            self._n_sub_steps = n_sub_steps
        elif control_timestep is not None:
            self._n_sub_steps = compute_n_steps(control_timestep,
                                            self._physics.timestep())
        else:
            self._n_sub_steps = 1

        # Add support for DEFAULT_TIME_LIMIT in env spec
        if time_limit is None and self.DEFAULT_TIME_LIMIT is None:
            time_limit = float('inf')
        elif time_limit is None:
            time_limit = self.DEFAULT_TIME_LIMIT

        # Default DM control time limit calculations
        if time_limit == float('inf'):
            self._step_limit = float('inf')
        else:
            self._step_limit = time_limit / (
                                self._physics.timestep() * self._n_sub_steps)
        self._step_count = 0
        
        self.seed()
        self.set_action_space()
        obs = self._get_obs()
        self.set_observation_space(obs)
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def set_action_space(self):
        if self.pad_actions:
            # Make Ac space really small for non-existant joints.
            high = np.ones(len(self._morphology) - 1) # actions correspond to edges
            high[self._morphology.joint_map == 0] = 1e-9 # Set to be very small.
            low = -1 * high
            self.action_space = gym.spaces.Box(low=low, high=high)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self._morphology.num_joints,))

    def set_observation_space(self, obs):
        spaces_dict = OrderedDict()
        for k, v in obs.items():
            if v is None:
                continue
            if k == 'edge_index':
                spaces_dict[k] = gym.spaces.Box(low=0, high=len(self._morphology), shape=v.shape, dtype=np.long)
            else:
                spaces_dict[k] = gym.spaces.Box(low=-np.inf, high=-np.inf, shape=v.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces_dict)

    def _post_step(self, action):
        '''
        Function designed to be overwritten for running the environment.
        MUST contain the following code:

        for _ in range(self._n_sub_steps):
            self._physics.step()

        Should most likley also call _get_obs()
        '''
        raise NotImplementedError
        
    def step(self, action):
        '''
        Wrapper around _step to handle time limit and grow / shrinking.
        '''
        if self.pad_actions:
            action = self._morphology.shrink(action)
        self._physics.set_control(action)
        try:
            for _ in range(self._n_sub_steps):
                self._physics.step()
        except:
            if self.allow_exceptions:
                pass
            else:
                raise
        obs, reward, done, info = self._post_step(action)
        self._step_count += 1
        done = done or self._step_count >= self._step_limit
        return obs, reward, done, info
    
    def render(self, mode='rgb_array', height=240, width=240, camera_id=0):
        # assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        img = self._physics.render(height=height, width=width, camera_id=camera_id)
        if mode == 'human':
            if self.viewer is None:
                from .viewer import OpenCVImageViewer
                self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            return img

    def _get_obs(self):
        # Shape of all observations (except for edge list) are num_nodes x num_features
        # Can easily be over written by adding more components to the dict or editting components in the dict.
        qpos = self._physics.data.qpos[-self._morphology.num_joints:].copy()
        qvel = self._physics.data.qvel[-self._morphology.num_joints:].copy()
        qpos, qvel = self._morphology.expand(qpos), self._morphology.expand(qvel)
        edge_attr = np.column_stack((qpos, qvel)).astype(np.float32)

        # Get position data relative to the "root" of the morphology
        if self.use_end_sites:
            xpos = self._physics.data.site_xpos[self.morphology.end_site_indices].copy()
        else:
            xpos = self._physics.data.xpos[-len(self._morphology):].copy()
        xpos[:, :2] -= xpos[0, :2] # Subtract root position.
        xvel = self._physics.data.subtree_linvel[-len(self._morphology):].copy()
        x = np.concatenate((xpos, xvel), axis=1).astype(np.float32)

        edges = self._morphology.edge_list
        obs = dict(x=x, edge_index=edges, edge_attr=edge_attr)
        u = self._get_task_obs()
        if not u is None:
            obs['u'] = u.astype(np.float32)
        return obs

    @staticmethod
    def get_morphology_obs(morphology, include_segments=False):
        edges = morphology.edge_list
        x = morphology.node_embeddings
        if include_segments:
            x = np.concatenate((x, morphology.segment_embeddings), axis=1)
        edge_attr = morphology.edge_embeddings
        return dict(x=x, edge_index=edges, edge_attr=edge_attr)

    def _get_task_obs(self):
        return None

    def _reset(self, **kwargs):
        '''
        Method to be overwritten for randomization etc. upon resets.
        See DM Control Suite Envs for examples of what to do in reset context with self._physics
        '''
        pass

    def reset(self, **kwargs):
        with self._physics.reset_context():
            self._reset(**kwargs)
        self._step_count = 0
        return self._get_obs()
