import gym
import numpy as np
from .base import MorphologyEnv

def to_node_graph(graph_obs, undirected=True):
    '''
    Removes all edge features and concatenates them with child node features.
    Also, changes graph to be undirected if specified. True by default.

    This function relies on the graph being a tree, and thus the root having no
    associated parent edge.
    '''
    if undirected:
        edge_index = np.concatenate((graph_obs['edge_index'], np.roll(graph_obs['edge_index'] , 1, axis=1)), axis=0)
    else:
        edge_index = graph_obs['edge_index']
    new_graph_features = [graph_obs['x']]
    if 'edge_attr' in graph_obs and not graph_obs['edge_attr'] is None:
        # Edges features: num_edges x F. The root does not have an edge, so we need to pad it.
        padding = np.zeros((1, graph_obs['edge_attr'].shape[1]))
        padded_edge_features = np.concatenate((padding, graph_obs['edge_attr']), axis=0)
        new_graph_features.append(padded_edge_features)
    if 'u' in graph_obs and not graph_obs['u'] is None:
        assert len(graph_obs['u'].shape) == 2, "Global features must include batch dim"
        u = np.tile(graph_obs['u'], (len(graph_obs['x']), 1))
        new_graph_features.append(u)
    x = np.concatenate(new_graph_features, axis=1)
    return dict(x=x, edge_index=edge_index)

def to_line_graph(graph_obs, line_edges=None):
    '''
    Function takes in an observation of a regular graph (directed or undirected edges)
    and returns the equivalent observation in line graph form (undirected edges)
    
    Note: This function _relies_ on all graphs passed in being trees.
    It also relies on edges being associated with the child. 
    So for edge (0, 1), the child in the tree is node 1. Each child has one
    node connected with its parent with an ID of node_id - 1.
    Example: The edge (3,6) is associated with node 6, and describes edge index 5.
    All of these paradigms are enforced by the morphology class.
    '''
    # If line edges are provided, we skip the computation to save the for loops.
    edges = graph_obs['edge_index']
    if line_edges is None:
        line_edges = []
        for i in range(len(edges) - 1):
            for j in range(i+1, len(edges)):
                # Consider all pairs of nodes (i,j) where i != j.
                node_set_1 = set(list(edges[i]))
                node_set_2 = set(list(edges[j]))
                if len(node_set_1.intersection(node_set_2)) > 0:
                    # These "body parts" share a connection.
                    # Need to update graph based on JOINT graph.
                    joint_id_1 = edges[i][1] - 1 # Get Child ID - 1
                    joint_id_2 = edges[j][1] - 1 # Get Child ID - 1
                    line_edges.append([joint_id_1 ,joint_id_2])
                    line_edges.append([joint_id_2, joint_id_1]) # Add both for undirected graph
        if len(line_edges) == 0:
            raise ValueError("Got zero line edges.")
        line_edges = np.array(line_edges)
    
    node_attr = graph_obs['x']
    parent_ids, child_ids = edges[:, 0], edges[:, 1]
    new_graph_features = [node_attr[parent_ids], node_attr[child_ids]]
    if 'edge_attr' in graph_obs and not graph_obs['edge_attr'] is None:
        new_graph_features.append(graph_obs['edge_attr'][child_ids - 1]) # The edge feature index is child_id - 1
    if 'u' in graph_obs and not graph_obs['u'] is None:
        assert len(graph_obs['u'].shape) == 2, "Global features must include a batch dim"
        u = np.tile(graph_obs['u'], (new_graph_features[0].shape[0], 1)) # Duplicate same number of times
    x = np.concatenate(new_graph_features, axis=1)
    return dict(x=x, edge_index=line_edges)

class DictObsWrapper(gym.Wrapper):
    '''
    Wrapper for extracting only some keys from the MorphologyEnv
    This is good for training on only edge features, for example.

    Note: this class does not override the static method get_morphology_obs(morphology).
    '''
    def __init__(self, env, keys=None):
        assert not keys is None, "Must provide a key to wrapper"
        super().__init__(env)
        self.keys = keys
        if len(self.keys) == 1:
            self.observation_space = self.env.observation_space[self.keys[0]]
        else:
            lows, highs = list(), list()
            for key in self.keys:
                lows.append(self.env.observation_space[key].low.flatten())
                highs.append(self.env.observation_space[key].high.flatten())
            low  = np.concatenate(lows, axis=0)
            high = np.concatenate(highs, axis=0)
            self.observation_space = gym.spaces.Box(low=low, high=high)

    def _wrap_obs(self, obs):
        if len(self.keys) == 1:
            return obs[self.keys[0]]
        else:
            out_obs = []
            for key in self.keys:
                out_obs.append(obs[key].flatten())
            out_obs = np.concatenate(out_obs, axis=0)
            return out_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._wrap_obs(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._wrap_obs(obs)

class NodeWrapper(gym.Wrapper):
    '''
    Wraps observations using to_node_graph
    '''

    def __init__(self, env):
        super().__init__(env)
        if not env.pad_actions:
            self.env.pad_actions = True
            self.env.set_action_space()
        self.action_space = self.env.action_space
        obs = to_node_graph(self.env._get_obs())
        self.env.set_observation_space(obs)
        self.observation_space = self.env.observation_space

    @staticmethod
    def get_morphology_obs(morphology, include_segments=False):
        return to_node_graph(MorphologyEnv.get_morphology_obs(morphology, include_segments=include_segments))

    def step(self, action):
        obs, reward, done, info = self.env.step(action[1:])
        return to_node_graph(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return to_node_graph(obs)

class LineGraphWrapper(gym.Wrapper):
    '''
    Wraps observations using to line graph using to_line_graph
    '''

    def __init__(self, env):
        super().__init__(env)
        if not env.pad_actions:
            self.env.pad_actions = True
            self.env.set_action_space()
        self.action_space = self.env.action_space
        obs = to_line_graph(self.env._get_obs(), line_edges=None)
        self._line_edges = obs['edge_index'] # Save edge index for later use.
        self.env.set_observation_space(obs)
        self.observation_space = self.env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return to_line_graph(obs, line_edges=self._line_edges), reward, done, info

    @staticmethod
    def get_morphology_obs(morphology, include_segments=False):
        return to_line_graph(MorphologyEnv.get_morphology_obs(morphology, include_segments=include_segments))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return to_line_graph(obs, line_edges=self._line_edges)
