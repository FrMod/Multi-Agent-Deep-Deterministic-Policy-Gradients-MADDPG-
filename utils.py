import torch
import numpy as np

def weights_init_uniform(m):
    f = 1/np.sqrt(m.weight.data.size()[0])
    torch.nn.init.uniform_(m.weight.data, -f, f)
    torch.nn.init.uniform_(m.bias.data, -f, f)