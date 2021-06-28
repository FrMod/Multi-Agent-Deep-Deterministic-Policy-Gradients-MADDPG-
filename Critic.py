import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import weights_init_uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import weights_init_uniform

class DDPG_Critic(nn.Module):
    
    def __init__(self, critic_lr, input_size, action_size, name, weight_decay, hidden_dim=(400,300), seed=54):
        super(DDPG_Critic, self).__init__()
        
        self. name = name
        self.seed = torch.manual_seed(seed)

        self.input_dims = input_size
        self.fc1_dims = hidden_dim[0]
        self.fc2_dims = hidden_dim[1]
        self.n_actions = action_size
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc1.apply(weights_init_uniform)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2.apply(weights_init_uniform)
        
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def _format(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        y = F.relu(self.action_value(action))
        z = F.relu(torch.add(x, y))
        z = self.q(z)

        return z

    def save(self):
        torch.save(self.state_dict(),self.name)
    
    def load(self,path):
        self.load_state_dict(torch.load(path+"/"+self.name))