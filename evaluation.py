import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
from recomDatasets import *
import scipy
from scipy import stats
import math

def hit_ratio(device, net, test_vec, dataset, stacking, k=10):                       #calculate hitRatio @ 10
    test_data = dataset(test_vec)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=101, shuffle=False)
    hit_num = 0.0
    ndcg_sum = 0.0
    net.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        if stacking:
            inputs[2] = inputs[2].to(device)
            inputs[3] = inputs[3].to(device)
        labels = labels.to(device)
        scores = net(inputs).view(-1).tolist()
        if (len(scores) - scipy.stats.rankdata(scores))[0] < k:
            hit_num += 1.0
            ndcg_sum += math.log(2)/math.log(2 + (len(scores) - scipy.stats.rankdata(scores))[0])
        if i % 1000 == 0:
            print('{}/{} iterations in this set has been tested'.format(i+1, len(test_loader)))
    hit_rat = hit_num / len(test_loader)
    ndcg = ndcg_sum / len(test_loader)
    print('The hit-ratio of this model at top-{} is {}'.format(k, hit_rat))
    print('The NDCG of this model at top-{} is {}'.format(k, ndcg))
    return hit_rat, ndcg