import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

b = [0.02,0.01,0.005,0.0025]
t = 2

a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

print(a)