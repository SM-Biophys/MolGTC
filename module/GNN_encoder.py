import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv 
import torch.nn.functional as F

class MoleculeEdgeEncoder(nn.Module):
    def __init__(self, emb_dim, edge_feat_dim=5):
        super(MoleculeEdgeEncoder, self).__init__()
        self.linear = nn.Linear(edge_feat_dim, emb_dim)


    def forward(self, edge_attr):
        out = self.linear(edge_attr.float())
        return out

class RGINE_Layer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(RGINE_Layer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
            nn.LayerNorm(out_feats)
        )

        self.conv = GINEConv(nn=self.mlp, train_eps=True)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        return h

class GNN_encoder(torch.nn.Module):
    def __init__(self, in_feats ,hid_feats, out_feats, layer_nums=3,edge_attr=5,g_drop_ratio=0.2):
        super(GNN_encoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.layer_nums = layer_nums
        self.drop_ratio = g_drop_ratio

        self.edge_encoder = MoleculeEdgeEncoder(emb_dim=hid_feats, edge_feat_dim=edge_attr)

        self.gin_layer_list = nn.ModuleList()

        self.feat_proj = nn.Linear(in_feats, hid_feats)

        for i in range(layer_nums):

            if i == layer_nums - 1:  
                self.gin_layer_list.append(RGINE_Layer(hid_feats, out_feats))

            else:  
                self.gin_layer_list.append(RGINE_Layer(hid_feats, hid_feats))

    def forward(self, x, edge_index, edge_attr):
        x = self.feat_proj(x)

        edge_embedding = self.edge_encoder(edge_attr)

        h = x
        for layer_index in range(self.layer_nums):
            h = self.gin_layer_list[layer_index](h, edge_index, edge_embedding)

            if layer_index < self.layer_nums - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)

        return h