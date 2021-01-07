import gym
from functools import partial
import numpy as np
import torch as th
from torch import nn
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import make_proba_distribution
from torch.distributions import Normal
import torch_geometric
from torch_geometric import nn as gnn

from optimal_agents.algs.graph_rollout_buffer import obs_to_graph, to_batch

class GSequential(nn.Module):

    def __init__(self, *layers, graph_conv_class=None):
        super(GSequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.graph_conv_class = graph_conv_class
    
    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            if isinstance(layer, self.graph_conv_class):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

class NodeACPolicy(BasePolicy):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'auto',
                 use_sde=False,
                 activation_fn: str = 'ReLU',
                 log_std_init: float = 0.0,
                 ortho_init: bool = True,
                 graph_conv_class = 'GraphConv',
                 rbf : int = 0,
                 bandwidth : float = 1,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in ADAM optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5
        
        super(NodeACPolicy, self).__init__(observation_space,
                                                action_space,
                                                device,
                                                features_extractor_class=None,
                                                features_extractor_kwargs=None,
                                                optimizer_class=optimizer_class,
                                                optimizer_kwargs=optimizer_kwargs,
                                                squash_output=False)
        
        # Now, build the network architecture.
        if net_arch is None:
            net_arch = [128, 128]
        self.net_arch = net_arch
        self.activation_fn = vars(nn)[activation_fn]
        
        # Determine specifics of the graph extractor. 
        self.graph_conv_class = vars(gnn)[graph_conv_class]
        
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init

        self._build(lr_schedule)
        
    def _build(self, lr_schedule):
        
        # If seperate value net, append layers to value net.
        extractor_net, action_net, value_net = [], [], []
        last_layer_dim = self.observation_space['x'].shape[1]

        for layer in self.net_arch:
            if isinstance(layer, int): 
                extractor_net.append(self.graph_conv_class(last_layer_dim, layer))
                extractor_net.append(self.activation_fn())
                last_layer_dim = layer
            else:
                assert isinstance(layer, dict)
                if 'pi' in layer:
                    # Build the pi_layers
                    for pi_layer in layer['pi']:
                        action_net.append(nn.Linear(last_pi_layer_dim, pi_layer))
                        action_net.append(self.activation_fn())
                        last_pi_layer_dim = pi_layer
                if 'vf' in layer:
                    # Build the pi_layers
                    for vf_layer in layer['vf']:
                        value_net.append(nn.Linear(last_vf_layer_dim, vf_layer))
                        value_net.append(self.activation_fn())
                        last_vf_layer_dim = vf_layer
                break
            
            last_pi_layer_dim, last_vf_layer_dim = last_layer_dim, last_layer_dim

        action_net.append(nn.Linear(last_pi_layer_dim, self.action_space.shape[0])) # Just get the number of outputs per node.
        value_net.append(nn.Linear(last_vf_layer_dim, 1))

        self.extractor_net = GSequential(*extractor_net, graph_conv_class=self.graph_conv_class)
        self.action_net = nn.Sequential(*action_net)
        self.value_net = nn.Sequential(*value_net)
        self.log_std = nn.Parameter(th.ones(1) * self.log_std_init, requires_grad=True)
        
        if self.ortho_init:
            module_gains = {
                self.action_net[-1]: 0.01,
                self.value_net: 1
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def _get_dist_and_value(self, obs):
        latent = self.extractor_net(obs.x, obs.edge_index, batch=obs.batch)
        values = self.value_net(latent)
        values = gnn.global_mean_pool(values, batch=obs.batch) # Evaluate the value using the avg pooling
        mean_actions = self.action_net(latent)
        action_std = th.ones_like(mean_actions) * self.log_std.exp()
        dist = Normal(mean_actions, action_std)
        return dist, values
    
    def forward(self, obs, deterministic: bool = False) -> Tuple[List, th.Tensor, th.Tensor]:
        dist, values = self._get_dist_and_value(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample() # Reparameterization trick with rsample
        log_prob = dist.log_prob(actions)
        log_prob = gnn.global_add_pool(log_prob, batch=obs.batch).sum(dim=-1) # used to be .unsqueeze
        obs.x = actions
        dlist = obs.to_data_list()
        actions = [g.x.squeeze(-1) for g in dlist]
        return actions, values, log_prob
    
    def evaluate_actions(self, obs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        dist, values = self._get_dist_and_value(obs)
        entropy = dist.entropy()
        log_prob = dist.log_prob(obs.y)
        log_prob = gnn.global_add_pool(log_prob, batch=obs.batch).sum(dim=-1) #.squeeze(-1)
        entropy = gnn.global_add_pool(entropy, batch=obs.batch).sum(dim=-1) #.squeeze(-1)
        return values, log_prob, entropy
    
    def _predict(self, observation, deterministic=False):
        pass # Implement to avoid ABC error

    def predict(self, observation, state=None, mask=None, deterministic=False):
        # Overwrite the existing method
        if isinstance(observation, list):
            vectorized_env = True
        else:
            vectorized_env = False
            observation = [observation]
        observation = to_batch(obs_to_graph(observation)).to(self.device)
        with th.no_grad():
            dist, values = self._get_dist_and_value(observation)
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.rsample() # Reparameterization trick with rsample
            observation.x = actions
            dlist = observation.to_data_list()
            actions = [g.x.squeeze(-1) for g in dlist]
        
        actions = [action.cpu().numpy() for action in actions]
        
        clipped_actions = actions
        # Clip the actions to avoid out of bound error when using gaussian distribution
        if isinstance(self.action_space, gym.spaces.Box) and not self.squash_output:
            clipped_actions = [np.clip(action, self.action_space.low, self.action_space.high) for action in actions]

        if not vectorized_env:
            clipped_actions = clipped_actions[0]

        return clipped_actions, None

    def value(self, observation):
        if isinstance(observation, list):
            vectorized_env = True
        else:
            vectorized_env = False
            observation = [observation]
        observation = to_batch(obs_to_graph(observation)).to(self.device)
        with th.no_grad():
            dist, values = self._get_dist_and_value(observation)
        if vectorized_env:
            values = values[0]
        return values.cpu().numpy()
    
    def _get_data(self) -> Dict[str, Any]:        
        data = super()._get_data()
        
        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            graph_conv_class=self.graph_conv_class,
            use_sde=self.use_sde,
            log_std_init=self.log_std_init,
            squash_output=self.dist_kwargs['squash_output'] if self.dist_kwargs else None,
            full_std=self.dist_kwargs['full_std'] if self.dist_kwargs else None,
            sde_net_arch=self.dist_kwargs['sde_net_arch'] if self.dist_kwargs else None,
            use_expln=self.dist_kwargs['use_expln'] if self.dist_kwargs else None,
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            ortho_init=self.ortho_init,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
        ))
        return data
