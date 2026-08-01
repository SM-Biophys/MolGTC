import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

from module.GNN_encoder import GNN_encoder
from module.NTXentloss import NTXentloss
from module.Transformer import MolecularTransformer


class GTCMol(nn.Module):
    def __init__(self,
                 hidden_dim=128,  
                 input_dim=120,  
                 edge_feature_dim=5,
                 gnn_layer_num=3,  
                 transformer_layers=4,
                 transformer_heads=8,
                 dropout_rate=0.1,
                 att_dropout_rate=0.1,
                 tau=0.1,
                 g_dropout_rate=0.2):
        super(GTCMol, self).__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau

        self.gnn_branch = GNN_encoder(
            in_feats=input_dim,
            hid_feats=hidden_dim * 2,
            out_feats=hidden_dim,
            layer_nums=gnn_layer_num,
            edge_attr=edge_feature_dim,
            g_drop_ratio=g_dropout_rate
        )

        self.transformer = MolecularTransformer(
            input_dim=input_dim,
            n_layers=transformer_layers,
            num_heads=transformer_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=att_dropout_rate
        )
        self.att_embeddings_proj = nn.Linear(hidden_dim, hidden_dim)

        self.contrast = NTXentloss(tau, alpha=2)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.gnn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.trans_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data):
        gnn_out = self.gnn_branch(data.x, data.edge_index, data.edge_attr)
        gnn_pool = global_mean_pool(gnn_out, data.batch)

        x_dense, mask = to_dense_batch(data.x, data.batch)

        trans_seq = self.transformer(x_dense, mask=mask)  
        trans_seq = self.att_embeddings_proj(trans_seq)  

        mask_float = mask.float().unsqueeze(-1)  

        sum_embed = (trans_seq * mask_float).sum(dim=1)  
        count = mask_float.sum(dim=1)  

        trans_pool = sum_embed / (count + 1e-6)  

        gnn_feat = self.gnn_proj(gnn_pool)
        trans_feat = self.trans_proj(trans_pool)


        final_embeds = self.fusion(torch.cat((gnn_feat, trans_feat), dim=1))

        loss_contrast = self.contrast(gnn_feat, trans_feat)

        return final_embeds, loss_contrast