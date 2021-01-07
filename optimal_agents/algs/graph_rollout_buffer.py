from typing import Generator, Optional, Union, List, NamedTuple
from stable_baselines3.common.vec_env import VecNormalize, VecEnv

import numpy as np
import torch as th
from gym import spaces
import torch_geometric

def obs_to_graph(obs, action=None):
    graphs = []
    for i in range(len(obs)):
        x, edge_index = th.from_numpy(obs[i]['x']).float(), th.from_numpy(obs[i]['edge_index'])
        if action is None:
            graphs.append(torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous()))
        else:
            ac = th.from_numpy(np.expand_dims(action[i], axis=-1)).float() # Ensure that action is shape Nodes X 1
            graphs.append(torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), y=ac))
    return graphs

def to_batch(obs: List):
    return torch_geometric.data.Batch.from_data_list(obs)

class GraphRolloutBufferSamples(NamedTuple):
    observations: torch_geometric.data.Batch
    # actions: torch_geometric.data.Batch
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class GraphRolloutBuffer(object):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (th.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """
    
    def __init__(self,
                 buffer_size: int,
                #  preprocessor,
                 device: Union[th.device, str] = 'cpu',
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 n_envs: int = 1):

        # self.preprocessor = preprocessor
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = [None for _ in range(self.buffer_size)] # EDIT: np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        # self.actions = [None for _ in range(self.buffer_size)] # EDIT: np.zeros((self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def compute_returns_and_advantage(self,
                                      last_value: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.
        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)
        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self,
            obs, # EDIT: Change obs to anything
            action, # EDIT: Change action to anything
            reward: np.ndarray,
            done: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (th.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (th.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        
        # self.observations[self.pos] = [preprocessor.preprocess_obs(single_obs) for sinlge_obs in obs]
        # self.actions[self.pos] = [preprocessor.preprocess_ac(single_obs, single_ac) for single_obs, single_ac in zip(self.observations[self.pos], action)]
        self.observations[self.pos] = obs_to_graph(obs, action=action)
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[GraphRolloutBufferSamples, None, None]:
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ['values', 'log_probs', 'advantages', 'returns']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            # EDIT: Flatten observations
            swapped_observations, swapped_actions = list(), list()
            num_envs = len(self.observations[0])
            for env_idx in range(num_envs):
                swapped_observations.extend([obs[env_idx] for obs in self.observations])
                # swapped_actions.extend([ac[env_idx] for ac in self.actions])
            self.observations = swapped_observations
            # self.actions = swapped_actions
            assert len(self.observations) == len(self.values)
            # assert len(self.actions) == len(self.values)
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None
               ):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[GraphRolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> GraphRolloutBufferSamples:
        observations = torch_geometric.data.Batch.from_data_list([self.observations[ind] for ind in batch_inds], follow_batch=['edge_attr']).to(self.device)
        # actions = torch_geometric.data.Batch.from_data_list([self.actions[ind] for ind in batch_inds], follow_batch=['edge_attr']).to(self.device)
        data = (
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())

        # return GraphRolloutBufferSamples(*((observations, actions) + tuple(map(self.to_torch, data))))
        return GraphRolloutBufferSamples(*((observations,) + tuple(map(self.to_torch, data))))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array: (np.ndarray)
        :param copy: (bool) Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: (th.Tensor)
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
