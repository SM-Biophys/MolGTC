import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  
        v = v.transpose(1, 2)  
        k = k.transpose(1, 2).transpose(2, 3)  

        q = q * self.scale
        x = torch.matmul(q, k)  

        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  

        x = x.transpose(1, 2).contiguous()  
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)  
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MolecularTransformer(nn.Module):
    """
    Fusion Transformer (GraphGPS Style)
    Input is a dense sequence with position and edge infocolumns。
    """

    def __init__(self,
                 input_dim,  
                 n_layers=4,
                 num_heads=8,
                 hidden_dim=256,  
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim  
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.n_layers = n_layers

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)


        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_dense, mask=None):
        """
        Input:
            x_dense: [Batch, Max_Nodes, 147]  (densecolumns，with padding)
            mask: [Batch, Max_Nodes] (True=valid, False=padding)
        """

        tensor = self.input_projection(x_dense)  


        attn_bias = None
        if mask is not None:
            key_padding_mask = ~mask

            attn_bias = torch.zeros(x_dense.size(0), 1, 1, x_dense.size(1), device=x_dense.device)

            attn_bias = attn_bias.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        for layer in self.layers:
            tensor = layer(tensor, attn_bias=attn_bias)

        tensor = self.final_norm(tensor)
        output = self.output_projection(tensor)  

        return output