import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_space: int, action_space: int, seed = None):
        super().__init__()
        if seed != None:
            torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, action_space)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x