# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class DiseaseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(DiseaseGNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        src = edge_index[0]
        dst = edge_index[1]
        edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.classifier(edge_repr)