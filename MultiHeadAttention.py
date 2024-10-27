import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dimensions,  game_indices = None , num_heads=8, dropout=0.1, relation_bias=False):
        super().__init__()

        embed_dim = dimensions[2]
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = dimensions[2]
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.relation_bias = relation_bias
        self.total_games = dimensions[1]
        self.game_indices = game_indices
        self.batch_size = dimensions[0]


        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        if self.relation_bias:
            # Changed to handle batching - will be expanded in forward pass
            self.bias = nn.Parameter(
                torch.zeros(self.batch_size, num_heads, self.total_games, self.total_games)
            )

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        seq_len = query.size(1) if len(query.size()) > 2 else 1



        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)



        if self.relation_bias:


            # Index the bias using game_indices and expand to match attention dimensions
            # game_indices shape: (batch_size,)
            B = self.bias  # Shape: (batch_size, num_heads, 1, 1)

            #print("selected bias shape" , selected_bias.shape)

            # Expand bias to match attention shape




            # Calculate attention scores with bias

            A1 = torch.matmul(q, k.transpose(-2, -1)) * self.scaling




            attn_weights = torch.add(A1, B)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        ).squeeze(1)

        output = self.out_proj(attn_output)

        return output


class RTransformer(nn.Module):
    def __init__(self, total_games, embed_dim=2304, num_heads=8, dropout=0.1):

        super().__init__()
        self.embed = embed_dim
        self.layer_norm = nn.LayerNorm([self.embed])
        self.MultiHeadAttention = MultiHeadAttention(total_games, self.embed, 8, relation_bias=True)
        self.MLPGate = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.Sigmoid(),

            )

    def forward(self, Matrix):
        ln = self.layer_norm(Matrix)
        rmha = self.MultiHeadAttention(ln, ln, ln)
        gate0 = self.MLPGate(rmha) #this is a gate

        final = self.layer_norm(gate0)

        return final