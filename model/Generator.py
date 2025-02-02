import torch
import numpy as np

import pandas as pd
import torch.nn as nn
from .network.GatedCrossAttention import GatedCrossAttention
from .network.SelfFusion import SelfFusion
from .network.SelfFusion import ConcatClassify
from .network.RGTransformer import RGTransformer
from .network.MultiHeadAttention import RTransformer

class Generator(nn.Module):
    def __init__(self, his_dimensions, k, employee_dim, total_dim_out,dropout=0.1):
        #k is the total games to generate
        super().__init__()

        self.total_games_out = total_dim_out
        self.employee_dim = employee_dim
        self.embed_dim = his_dimensions[2]
        self.total_games_his = his_dimensions[1]
        self.total_games = self.total_games_his + k
        self.k = k
        self.batch_size = his_dimensions[0]

        self.RGThis = RGTransformer(his_dimensions  , dropout=dropout)
        self.RGTgen = RGTransformer((self.batch_size, k, self.embed_dim),  dropout=dropout)
        self.GCA = GatedCrossAttention((self.batch_size ,self.total_games , self.embed_dim))
        self.SF = SelfFusion((self.batch_size , k , self.embed_dim))
        self.CC = ConcatClassify(employee_dim,employee_dim ,total_dim_out )

    def forward(self, employee, historical, generated):
        #employee is the employee embedding of shape (1, employee_dim)
        #historical is the historical game embeddings of shape (total_games_his, embed_dim)
        #generated is the generated game embeddings of shape (k, embed_dim)
        RGThis = self.RGThis(historical)
        RGTgen = self.RGTgen(generated)
        gsa_embedding = self.GCA(RGTgen , RGThis)
        self_fusion_embedding = self.SF(gsa_embedding)
        out = self.CC(employee, self_fusion_embedding)
        return out



# class Discriminator(nn.Module):
#
#     def __init__(self, total_games, embed_dim, employee_dim, dropout=0.1):
#         super().__init__()
#         self.total_games = total_games
#         self.embed_dim = embed_dim
#         self.employee_dim = employee_dim
#
#
#         self.SF = SelfFusion(total_games, embed_dim, dropout=dropout)
#         self.CC = ConcatClassify(employee_dim, embed_dim, 1, generator=False, dropout=dropout)
#         self.RT = RTransformer(total_games, embed_dim, dropout=dropout)
#
#     def forward(self, employee, historical):
#         #employee is the employee embedding of shape (1, employee_dim)
#         #historical is the historical game embeddings of shape (total_games, embed_dim)
#         #generated is the generated game embeddings of shape (k, embed_dim)
#
#         rt = self.RT(historical)
#         self_fusion_embedding = self.SF(rt)
#         out = self.CC(employee, self_fusion_embedding)
#         return out



    # game_history = torch.load('game_embeddings.pt', weights_only=False)
    # print(game_history)
    #get the historical game embeddings

    #historical_game_embeddings = data.iloc[:, 100:200].values