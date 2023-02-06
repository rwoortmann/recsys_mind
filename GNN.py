from torch_geometric.nn import LayerNorm, GraphNorm, InstanceNorm
from torch_geometric.utils import degree
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.act = nn.Tanh()
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x

    

###
# Layers
###

class FirstConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.FFN = MLP(in_channels, hidden_channels, out_channels)
        
        
    def forward(self, subgraph_feats, edge_index, atr):
        n_neighbors = degree(edge_index[atr].storage.col(), num_nodes=edge_index[atr].storage._sparse_sizes[1])
        agr_feats = edge_index[atr].t() @ subgraph_feats['news']
        agr_feats = agr_feats / (n_neighbors[:,None] + 1e-08)
        
        out = self.FFN(agr_feats)
        return out
        


class NewsConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.FFN = MLP(in_channels, hidden_channels, out_channels)
        
    def forward(self, subgraph_feats, edge_index, atr_nodes):
        out = subgraph_feats['news']
        
        for atr in atr_nodes:
            n_neighbors = degree(edge_index[atr].storage.row(), num_nodes=edge_index[atr].storage._sparse_sizes[0])
            agr_feats = edge_index[atr] @ subgraph_feats[atr]
            agr_feats = agr_feats / (n_neighbors[:,None] + 1e-08)
            out = torch.cat((out, agr_feats), dim=1)
        out = self.FFN(out)
        return out
        


class UserConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.FFN = MLP(in_channels, hidden_channels, out_channels)
        
    def forward(self, subgraph_feats, edge_index):
        n_neighbors = degree(edge_index['users'].storage.col(), num_nodes=edge_index['users'].storage._sparse_sizes[1])
        agr_feats = edge_index['users'].t() @ subgraph_feats['news']
        agr_feats = agr_feats / (n_neighbors[:,None] + 1e-08)
    
        out = torch.cat((subgraph_feats['users'], agr_feats), dim=1)
        out = self.FFN(out)
        return out

    

## Net    
class GCN(nn.Module):
    def __init__(self, news_dim, empty_nodes=[], atr_nodes=[]):
        super().__init__()
        in_dim = news_dim
        hidden_dim = 256
        out_dim = 128
        self.empty_nodes = empty_nodes
        self.atr_nodes = atr_nodes
                
        self.first_conv = nn.ModuleList([FirstConv(in_dim, hidden_dim, out_dim) for atr in self.empty_nodes])
        self.news_conv = NewsConv(in_dim + len(empty_nodes)*out_dim + len(set(self.atr_nodes)-set(self.empty_nodes))*100, hidden_dim*2, out_dim)
        self.user_conv = UserConv(out_dim*2, hidden_dim, out_dim)
        
    def forward(self, news_x, edge_index):
        #subgraph_feats = {atr: data.features[atr] for atr in set(self.atr_nodes)-set(self.empty_nodes)}
        subgraph_feats = {'news': news_x}
        print(news_x.shape)
        # prop news feats to empty atribute nodes
        for i, atr in enumerate(self.empty_nodes):
            subgraph_feats[atr] = self.first_conv[i](subgraph_feats, edge_index, atr)

        # update news nodes
        subgraph_feats['news'] = self.news_conv(subgraph_feats, edge_index, self.atr_nodes)
        
        # update atr nodes
        subgraph_feats['users'] = self.user_conv(subgraph_feats, edge_index)
        print(subgraph_feats['news'].shape)
        print(subgraph_feats['users'].shape)
        
        print()                
        return subgraph_feats








