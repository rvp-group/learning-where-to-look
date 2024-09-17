import torch.nn as nn
from lwl.apps.utils.seed import *

N_DIM=300

class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, modality='classification'):
        super(MLPClassifier, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, N_DIM)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(N_DIM, N_DIM)
        self.relu2 = nn.ReLU()
        self.fcfinal = nn.Linear(N_DIM, output_size)
        self.modality = modality
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fcfinal(x)
        x = self.sigmoid(x)
        return x