import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric

from optimal_agents.utils.loader import get_env, get_morphology

from optimal_agents.algs.graph_rollout_buffer import obs_to_graph
from optimal_agents.morphology import Morphology

class NodeMorphologyVF(nn.Module):

    def __init__(self, params, lr=0.001, batch_size=64, buffer_size=3072, include_segments=False, 
                       thompson=False):
        super().__init__()
        self.params = params
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.thompson = thompson
        self.include_segments = include_segments

        if 'activation_fn' in self.params['policy_kwargs']:
            self.act_fn = vars(F)[self.params['policy_kwargs']['act_fn']]
        else:
            self.act_fn = torch.tanh
        if 'net_arch' in self.params['policy_kwargs']:
            net_arch = self.params['policy_kwargs']['net_arch']
        else:
            net_arch = [128, 128, 128]
        if 'graph_conv_class' in self.params['policy_kwargs']:
            graph_layer = vars(torch_geometric.nn)[self.params['policy_kwargs']['graph_conv_class']]
        else:
            graph_layer = torch_geometric.nn.GraphConv

        # Construct a test environment to get the input / output sizes
        test_morph = get_morphology(params)
        test_env = get_env(params, morphology=test_morph)
        self.morphology_graph_fn = test_env.get_morphology_obs # Make srue we have the correct conversion function.
        morphology_obs = self.morphology_graph_fn(test_morph, self.include_segments)
        
        last_layer_dim = morphology_obs['x'].shape[1] # Get the input size for each node.

        layers = []
        for layer in net_arch:
            layers.append(graph_layer(last_layer_dim, layer))
            last_layer_dim = layer
        self.last_extractor_dim = last_layer_dim
        self.layers = nn.ModuleList(layers)
        self.linear = torch.nn.Linear(last_layer_dim, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.data = []
        self.test_dropout_mask = None

        test_env.close()
        del test_env # Force close this.

    def forward(self, graph, dropout_mask=None):
        x, edge_index = graph.x, graph.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.act_fn(x)

        x = torch_geometric.nn.global_mean_pool(x, graph.batch) # Global pool before linear layer to support thompson dropout

        if self.thompson:
            if dropout_mask is None:
                dropout_mask = torch.distributions.Bernoulli(torch.full_like(x, 0.5)).sample() / 0.5
            x = x * dropout_mask

        x = self.linear(x)
        return x

    def update(self, data, n_epochs=8):
        self.test_dropout_mask = None # Set the dropout mask to None. We are no longer in the same assessment
        self.data.extend(data)
        if len(self.data) > self.buffer_size:
            num_over = len(self.data) - self.buffer_size
            del self.data[:num_over] # Remove the first num_over elements from the buffer

        # Normalize the data for the update.
        reward_values = [data_pt[1] for data_pt in self.data]
        reward_mean = np.mean(reward_values)
        reward_std = np.std(reward_values)

        for epoch in range(n_epochs):
            perm = np.random.permutation(len(self.data))
            num_full_batches = len(perm) // self.batch_size
            for i in range(num_full_batches + 1):
                if i != num_full_batches:
                    inds = perm[i*self.batch_size:(i+1)*self.batch_size]
                else:
                    inds = perm[i*self.batch_size:]
                if len(inds) == 0:
                    continue
                y = torch.from_numpy(np.array([(self.data[ind][1] - reward_mean)/reward_std for ind in inds])).float().unsqueeze(-1)
                graph = torch_geometric.data.Batch.from_data_list([self.data[ind][0] for ind in inds])
                self.optim.zero_grad()
                values = self.forward(graph)
                assert values.shape == y.shape
                loss = self.criterion(values, y)
                loss.backward()
                self.optim.step()

        # Test Loss
        with torch.no_grad():
            print(len(self.data), "DATA PTS")
            graph = torch_geometric.data.Batch.from_data_list([self.data[ind][0] for ind in range(len(self.data))])
            y = torch.from_numpy(np.array([(self.data[i][1] - reward_mean)/reward_std for i in range(len(self.data))])).float()
            test_loss = self.criterion(self.forward(graph), y.unsqueeze(-1)).item()
        return test_loss
   
    def evaluate(self, morphology):
        if self.thompson and self.test_dropout_mask is None:
            self.test_dropout_mask = (torch.distributions.Bernoulli(
                                        torch.full((1, self.last_extractor_dim), 0.5)
                                     ).sample() / 0.5)
        graph = self.morphology_graph_fn(morphology, include_segments=self.include_segments)
        graph = torch_geometric.data.Batch.from_data_list(obs_to_graph([graph]))
        with torch.no_grad():
            value = self.forward(graph, dropout_mask=self.test_dropout_mask)[0]
        return value
