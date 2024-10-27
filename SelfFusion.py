import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter

from MultiHeadAttention import MultiHeadAttention

class SelfFusion(nn.Module):
    def __init__(self , dimensions, dropout=0.1):
        super().__init__()

        total_games = dimensions[1]
        print("dimensionsf" , dimensions)
        embed_dim = dimensions[2]

        self.total_games = total_games
        self.embed = embed_dim
        batch_size = dimensions[0]
        self.batch_size = batch_size
        print("sdafafsferf" , batch_size)

        self.layer_norm = nn.LayerNorm([embed_dim])

        self.MultiHeadAttention = MultiHeadAttention(dimensions , game_indices=[0]*batch_size)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),

        )
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.Wg = nn.Parameter(
            torch.randn(embed_dim, embed_dim)
        )
        self.Wa = nn.Parameter(
            torch.randn(embed_dim, embed_dim)
        )
        self.va = nn.Parameter(
            torch.randn(total_games, 1)
        )

        self.Wp = nn.Parameter(
            torch.randn(self.total_games*self.embed, 449)
        )

    def forward(self, gsa_embedding):
        print("gsa_embedding" , gsa_embedding.shape)
        t = self.layer_norm( gsa_embedding + self.MultiHeadAttention(gsa_embedding, gsa_embedding, gsa_embedding) )
        t1 = self.layer_norm(t + self.MLP(t))
        print("t1" , t1.shape , "Wa", self.Wa.shape , "va", self.va.shape )
        t2 = self.tanh(torch.matmul(t1, self.Wa) + self.va)

        print("t2" , t2.shape)
        t3 = self.softmax(torch.matmul(t2, self.Wg))
        print("t3", t3.shape)
        #reduce t3 to 3 dimensions
        t3 = t3.reshape(self.batch_size, -1)
        print("t3", t3.shape)

        #t4 = torch.matmul(t3, t1)

        final = torch.matmul(t3 , self.Wp)
        return final


class ConcatClassify(nn.Module):
    def __init__(self, employee_dim, self_fusion_dim, total_games_out = 2304, generator=True, dropout=0.1):
        super().__init__()

        print("employee_dim" , employee_dim , "self_fusion_dim" , self_fusion_dim)

        if generator:
            self.linear = nn.Sequential(
                nn.Linear(employee_dim + self_fusion_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, total_games_out),
                nn.ReLU()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(employee_dim + self_fusion_dim, 100),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(100, 1),
                nn.Sigmoid()
            )

    def forward(self, user_embed, self_fusion_embedding):


        x = torch.cat([user_embed, self_fusion_embedding], dim=1)
        print("x" , x.shape)
        return self.linear(x)


if __name__ == '__main__':
    gen_em = torch.load('./combined_concatenated_game_embeddings.pt', weights_only=False)


    gen = []

    for i in range(len(gen_em)):
        gen.append(gen_em[i][1])


    gen = np.array(gen)

    total_games , embed_dim = gen.shape[0], gen.shape[1]
    sf = SelfFusion( (1,gen.shape[0],gen.shape[1]) , dropout=0.1 )
    out = sf(torch.from_numpy(gen).float() )

    print(out)

    cc = ConcatClassify( embed_dim, embed_dim , 20)

    out = cc(out, out)

    print(out)