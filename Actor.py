import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque

import matplotlib.pylab as plt
import copy
import numpy as np

from utils import weights_init_uniform


class DDPG_Actor(nn.Module):

    def __init__(self, actor_lr, input_size, action_size, name,  hidden_dim=(400,300), seed=55):
        super(DDPG_Actor, self).__init__()

        self. name = name
        self.seed = torch.manual_seed(seed)   

        self.input_dims = input_size
        self.fc1_dims = hidden_dim[0]
        self.fc2_dims = hidden_dim[1]
        self.n_actions = action_size
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc1.apply(weights_init_uniform)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2.apply(weights_init_uniform)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)     
        
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
    
    def forward(self, x):   
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save(self):
        torch.save(self.state_dict(),self.name)
    
    def load(self,path):
        self.load_state_dict(torch.load(path+"/"+self.name))