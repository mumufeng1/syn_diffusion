from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
from IPython.display import display
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.optim import Adam
from tqdm import tqdm_notebook
from torchvision.utils import save_image
import matplotlib
import math
from inspect import isfunction
from functools import partial
import scipy
from scipy.special import rel_entr
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.image as mpimg
import glob
from PIL import Image
from itertools import product
import random


class train_dataset(Dataset):
    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, ix):
        self.fea = self.feature[ix][:]
        data = {}
        data['feature'] = torch.from_numpy(np.array(self.fea, dtype=float)).type(torch.FloatTensor)
        return data

    def __len__(self):
        return len(self.feature)


def seq2onehot(seq, seq_len):
    module = np.array([[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]])
    i = 0
    promoter_onehot = []
    while i < len(seq):
        tmp = []
        for item in seq[i]:
            if item == 't' or item == 'T':
                tmp.append(module[0])
            elif item == 'c' or item == 'C':
                tmp.append(module[1])
            elif item == 'g' or item == 'G':
                tmp.append(module[2])
            elif item == 'a' or item == 'A':
                tmp.append(module[3])
            else:
                tmp.append([0, 0, 0, 0])
        promoter_onehot.append(tmp)
        i = i + 1
    data = np.zeros((len(seq), seq_len, 1, 4))
    data = np.float32(data)
    m = 0
    while m < len(seq):
        n = 0
        while n < seq_len:
            data[m, n, 0, :] = promoter_onehot[m][n]
            n = n + 1
        m = m + 1
    return data