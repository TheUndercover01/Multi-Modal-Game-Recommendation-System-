import torch
import numpy as np
import torch.nn as nn

from .MultiHeadAttention import MultiHeadAttention
class GatedCrossAttention(nn.Module):
    '''gated cross attention
    issue with total games as the total num of games are not the same in his and gen. what is total_game in thsi his+gen or something else we need to change in future

    '''
    def __init__(self, dimensions, dropout=0.1):
        super().__init__()

        total_games = dimensions[1]
        embed_dim = dimensions[2]
        batch_size = dimensions[0]

        self.layer_norm = nn.LayerNorm([embed_dim])

        self.MultiHeadAttention = MultiHeadAttention( dimensions, game_indices=[0]*batch_size )
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            #nn.Sigmoid(),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, generated, historical):
        S_t = self.layer_norm(generated)

        l = self.layer_norm(historical)
        S_t_1 = self.MultiHeadAttention(S_t, l, l)
        S_t_2 = torch.mul(S_t_1, self.sigmoid(self.MLP(S_t_1))) + S_t

        gca = torch.mul(self.sigmoid( self.MLP(S_t_2)), self.layer_norm(self.MLP(S_t_2))) + S_t_2
        return gca



if __name__ == '__main__':
    gen_em = torch.load('../../combined_concatenated_game_embeddings.pt', weights_only=False)
    his_em = torch.load('../../combined_concatenated_game_embeddings.pt', weights_only=False)

    gen = []
    his = []
    for i in range(len(gen_em)):
        gen.append(gen_em[i][1])
        his.append(his_em[i][1])

    gen = np.array(gen)
    his = np.array(his)
    total_games , embed_dim = gen.shape[0], gen.shape[1]
    gca = GatedCrossAttention(total_games, embed_dim )
    out = gca(torch.from_numpy(gen).float() , torch.from_numpy(his).float())



