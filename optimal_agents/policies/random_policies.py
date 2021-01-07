import itertools
import numpy as np

'''
Random action primitive policies used when collecting data with TAME.
'''

class CosinePolicy(object):

    def __init__(self, action_dim, sample_freq=50, num_freqs=2, num_phases=2):
        self.action_dim = action_dim
        self.period = sample_freq
        freqs = [2*np.pi / self.period * (2**i) for i in range(num_freqs)]
        phases = [2*np.pi*i / num_phases for i in range(num_phases)]
        self.action_set = np.array(list(itertools.product(freqs, phases)))
        self._num_actions = len(self.action_set)

    @property
    def num_actions(self):
        return self._num_actions

    def step(self, num_steps):
        labels = np.random.randint(low=0, high=len(self.action_set), size=(self.action_dim))
        actions = self.action_set[labels]
        freq = np.expand_dims(actions[:,0], axis=0)
        phase = np.expand_dims(actions[:,1], axis=0)
        x = np.expand_dims(np.arange(0, num_steps), axis=1)
        actions = np.cos(freq*x + phase)
        return actions, labels

class GlobalCosinePolicy(object):
    def __init__(self, action_dim, sample_freq=50, num_freqs=2, num_phases=2):
        self.action_dim = action_dim
        self.period = sample_freq
        freqs = [2*np.pi / self.period * (2**i) for i in range(num_freqs)]
        phases = [2*np.pi*i / num_phases for i in range(num_phases)]
        self.action_set = np.array(list(itertools.product(freqs, phases)))
        self._num_actions = len(self.action_set)

    @property
    def num_actions(self):
        return self._num_actions

    def step(self, num_steps):
        label = np.random.randint(low=0, high=len(self.action_set))
        labels = np.array([label for _ in range(self.action_dim)])
        actions = self.action_set[labels]
        freq = np.expand_dims(actions[:,0], axis=0)
        phase = np.expand_dims(actions[:,1], axis=0)
        x = np.expand_dims(np.arange(0, num_steps), axis=1)
        actions = np.cos(freq*x + phase)
        return actions, labels

class BinaryPolicy(object):
    def __init__(self, action_dim, **kwargs): # allow for throw away kwargs
        self.action_set = [0, 1]
        self.action_dim = action_dim
    
    @property
    def num_actions(self):
        return 2 # its binary pls

    def step(self, num_steps):
        labels = np.random.randint(2, size=(self.action_dim)) # generates random bits
        actions = 2*labels - 1
        actions = np.tile(actions, (num_steps, 1))
        return actions, labels

class NaryPolicy(object):
    def __init__(self, action_dim, num_freqs=4, **kwargs):
        assert num_freqs % 2 == 0
        top_acs = np.array([1 - i/(num_freqs//2) for i in range(num_freqs // 2)])
        self.action_set = np.concatenate((-1*top_acs, top_acs), axis=0)
        self.action_dim = action_dim
        self._num_actions = len(self.action_set)

    @property
    def num_actions(self):
        return self._num_actions

    def step(self, num_steps):
        labels = np.random.randint(low=0, high=len(self.action_set), size=(self.action_dim))
        actions = self.action_set[labels]
        actions = np.tile(actions, (num_steps, 1))
        return actions, labels

class GlobalNaryPolicy(object):

    def __init__(self, action_dim, num_freqs=4, **kwargs):
        assert num_freqs % 2 == 0
        top_acs = np.array([1 - i/(num_freqs//2) for i in range(num_freqs // 2)])
        self.action_set = np.concatenate((-1*top_acs, top_acs), axis=0)
        self.action_dim = action_dim
        self._num_actions = len(self.action_set)

    @property
    def num_actions(self):
        return self._num_actions

    def step(self, num_steps):
        label = np.random.randint(low=0, high=len(self.action_set))
        labels = np.array([label for _ in range(self.action_dim)])
        actions = self.action_set[labels]
        actions = np.tile(actions, (num_steps, 1))
        return actions, labels
