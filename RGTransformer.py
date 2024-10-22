import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
embedding = torch.load('./combined_concatenated_game_embeddings.pt', weights_only=False)


class RelationMultiHeadAttention(nn.Module):
    def __init__(self, total_games, embed_dim=2304, num_heads=8, dropout=0.1):
        super().__init__()


        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.relation_bias = nn.Parameter(
           torch.zeros(total_games,num_heads,1,1)
        )

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, seq_len, embed_dim)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        #B is a bias matri
        B = self.relation_bias




        #print("k",k.shape)
        #print("B",B.shape)

        # Scaled dot-product attention
        A1 = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = torch.add(A1, B)


        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.embed_dim
        )


        output = self.out_proj(attn_output)

        return output


class RGTransformer(nn.Module):
    def __init__(self, game_total, shape1, dropout=0.1):
        #making a linear norm palayer
        super().__init__()
        self.embed = shape1

        #print(self.embed , self.shape0)


        self.layer_norm = nn.LayerNorm([self.embed])
        self.MultiHeadAttention = RelationMultiHeadAttention(game_total, self.embed, 8)
        self.MLPGate = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.Softmax(),

        )
        self.MLP = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.ReLU(),

        )
    def forward(self, Matrix):

        ln = self.layer_norm(Matrix)
        rmha = self.MultiHeadAttention(ln, ln, ln)
        gate0 = self.MLPGate(Matrix) #this is a gate

        gate1 = torch.mul(gate0, rmha) + Matrix


        final = torch.mul(gate1, self.layer_norm(self.MLP(gate1))) + gate1
        print("final shape" , final.shape)
        return final




if __name__ == '__main__':

    matrix = []
    for i in range(len(embedding)):
        matrix.append(embedding[i][1])

    matrix = np.array(matrix)
    game_total , embed = matrix.shape

    rgt = RGTransformer(game_total , embed)

    tensor_data = torch.from_numpy(matrix).float()

    print(rgt(tensor_data))




        #self.norm_layer = NormLayer(matrix.shape[0])

        #making a linear RG layer
