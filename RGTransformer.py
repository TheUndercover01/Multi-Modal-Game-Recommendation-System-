import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttention
embedding = torch.load('./combined_concatenated_game_embeddings.pt', weights_only=False)


class RGTransformer(nn.Module):
    def __init__(self, game_total, shape1, dropout=0.1):
        #making a linear norm palayer
        super().__init__()
        self.embed = shape1

        #print(self.embed , self.shape0)


        self.layer_norm = nn.LayerNorm([self.embed])
        self.MultiHeadAttention = MultiHeadAttention(game_total, self.embed, 8 , relation_bias=True)
        self.MLPGate = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.Sigmoid(),

        )
        self.MLP = nn.Sequential(
            nn.Linear(self.embed, self.embed),
            nn.Sigmoid(),

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
