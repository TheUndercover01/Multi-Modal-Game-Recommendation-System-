import torch
import numpy as np
import torch.nn as nn
from GatedCrossAttention import GatedCrossAttention
from SelfFusion import SelfFusion
from ConcatClassify import ConcatClassify
from RGTransformer import RGTransformer
from MultiHeadAttention import RTransformer

class Generator(nn.Module):
    def __init__(self, total_games_his, k, embed_dim,employee_dim, total_games_out,dropout=0.1):
        #k is the total games to generate
        super().__init__()

        self.total_games_out = total_games_out
        self.employee_dim = employee_dim
        self.embed_dim = embed_dim
        self.total_games_his = total_games_his
        self.k = k
        self.RGThis = RGTransformer(total_games_his,  dropout=dropout)
        self.RGTgen = RGTransformer(k,  dropout=dropout)
        self.GCA = GatedCrossAttention(k + total_games_his , embed_dim)
        self.SF = SelfFusion(embed_dim)
        self.CC = ConcatClassify(employee_dim,embed_dim ,total_games_out )

    def forward(self, employee, historical, generated):
        #employee is the employee embedding of shape (1, employee_dim)
        #historical is the historical game embeddings of shape (total_games_his, embed_dim)
        #generated is the generated game embeddings of shape (k, embed_dim)
        RGThis = self.RGThis(historical)
        RGTgen = self.RGTgen(generated)
        gsa_embedding = self.GCA(RGThis, RGTgen)
        self_fusion_embedding = self.SF(gsa_embedding)
        out = self.CC(employee, self_fusion_embedding)
        return out



class Discriminator(nn.Module):

    def __init__(self, total_games, embed_dim, employee_dim, dropout=0.1):
        super().__init__()
        self.total_games = total_games
        self.embed_dim = embed_dim
        self.employee_dim = employee_dim


        self.SF = SelfFusion(total_games, embed_dim, dropout=dropout)
        self.CC = ConcatClassify(employee_dim, embed_dim, 1, generator=False, dropout=dropout)
        self.RT = RTransformer(total_games, embed_dim, dropout=dropout)

    def forward(self, employee, historical):
        #employee is the employee embedding of shape (1, employee_dim)
        #historical is the historical game embeddings of shape (total_games, embed_dim)
        #generated is the generated game embeddings of shape (k, embed_dim)

        rt = self.RT(historical)
        self_fusion_embedding = self.SF(rt)
        out = self.CC(employee, self_fusion_embedding)
        return out