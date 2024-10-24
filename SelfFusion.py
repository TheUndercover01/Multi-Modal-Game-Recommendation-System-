import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter

from MultiHeadAttention import MultiHeadAttention

class SelfFusion(nn.Module):
    def __init__(self , total_games, embed_dim, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm([embed_dim])

        self.MultiHeadAttention = MultiHeadAttention(total_games,embed_dim)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),

        )
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.Wg = nn.Parameter(
            torch.randn(embed_dim, 1)
        )
        self.Wa = nn.Parameter(
            torch.randn(embed_dim, embed_dim)
        )
        self.va = nn.Parameter(
            torch.randn(total_games, 1)
        )

    def forward(self, gsa_embedding):
        t = self.layer_norm( gsa_embedding + self.MultiHeadAttention(gsa_embedding, gsa_embedding, gsa_embedding) )
        t1 = self.layer_norm(t + self.MLP(t))
        print(t1.shape)
        t2 = self.tanh(torch.matmul(t1, self.Wa) + self.va)
        t3 = self.softmax(torch.matmul(t2, self.Wg))
        print(t3.T.shape)
        final = torch.matmul(t3.T, t1)
        return final


class ConcatClassify(nn.Module):
    def __init__(self, employee_dim, self_fusion_dim, total_games_out, generator=True, dropout=0.1):
        super().__init__()

        if generator:
            self.linear = nn.Sequential(
                nn.Linear(employee_dim + self_fusion_dim, 100),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(100, total_games_out),
                nn.Softmax()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(employee_dim + self_fusion_dim, 100),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(100, 1),
                nn.Sigmoid()
            )

    def forward(self, gsa_embedding, self_fusion_embedding):


        x = torch.cat([gsa_embedding, self_fusion_embedding], dim=1)
        return self.linear(x)


if __name__ == '__main__':
    gen_em = torch.load('./combined_concatenated_game_embeddings.pt', weights_only=False)


    gen = []

    for i in range(len(gen_em)):
        gen.append(gen_em[i][1])


    gen = np.array(gen)

    total_games , embed_dim = gen.shape[0], gen.shape[1]
    sf = SelfFusion( total_games , embed_dim )
    out = sf(torch.from_numpy(gen).float() )

    print(out)

    cc = ConcatClassify( embed_dim, embed_dim , 20)

    out = cc(out, out)

    print(out)