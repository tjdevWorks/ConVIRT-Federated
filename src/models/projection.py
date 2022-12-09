import torch

class Projector(torch.nn.Module):
    def __init__(self, in_dim:int, hidden_size:int, out_dim:int):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden_size, bias=False)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_dim, bias=False)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.l2(x)

        return x