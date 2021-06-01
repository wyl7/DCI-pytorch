import torch
import torch.nn as nn
from layers import GraphCNN, AvgReadout, Discriminator
import sys
sys.path.append("models/")

class DGI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        super(DGI, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, lbl):
        criterion = nn.BCEWithLogitsLoss()
        h_1 = torch.unsqueeze(self.gin(seq1, adj), 0)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = torch.unsqueeze(self.gin(seq2, adj), 0)
        
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        loss = criterion(ret, lbl)

        return loss