import torch
import torch_geometric

class GraphConvModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, net_arch=[192, 192, 192]):
        super().__init__()
        layers = []
        last_layer_dim = in_dim
        for layer_dim in net_arch:
            layers.append(torch_geometric.nn.GraphConv(last_layer_dim, layer_dim))
            last_layer_dim = layer_dim
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(last_layer_dim, out_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.nn.functional.relu(x)
        x = self.linear(x)
        return x

''' For fully connected version of TAME 
    Currently not in this codebase, working on porting over.
'''

class MLP(torch.nn.Module):
    
    def __init__(self, in_dim, out_dim, layers=[128, 128], act='ReLU'):
        super().__init__()
        last_dim = in_dim
        model = []
        act = vars(torch.nn)[act]
        for layer in layers:
            model.append(torch.nn.Linear(last_dim, layer))
            last_dim = layer
            model.append(act())
        model.append(torch.nn.Linear(last_dim, out_dim))
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class MLP_Local(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_heads, net_arch=[256, 256]):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp = MLP(in_dim, num_classes*num_heads, layers=net_arch)

    def forward(self, x, edge_index, edge_attr, batch):
        # We only care about x
        x = self.mlp(x)
        return x.view(-1, self.num_classes, self.num_heads)
