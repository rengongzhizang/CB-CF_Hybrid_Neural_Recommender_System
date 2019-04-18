import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AutoEncoderDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
         temp = torch.from_numpy(self.x[idx]).type(torch.FloatTensor).view(-1)
         sample = (temp, temp)
         return sample

class GenresDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        temp = self.x[idx,:].view(-1)
        sample = (temp, temp)
        return sample

class OverviewDataset(Dataset):
    def __init__(self, x_train):
        self.x = x_train
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        temp = self.x[idx,:].view(-1)
        sample = (temp, temp)
        return sample

class RecsysDataset(Dataset):
    def __init__(self, inputs):
        self.user_feature = inputs[0]                        # user_feature = user_onehot
        self.item_feature = inputs[1]                        # item_feature = (item_onehot, item_overview)
        self.labels = inputs[2].view(-1)

        self.count = inputs[0].shape[0]

    def __getitem__(self, idx):
        features = (self.user_feature[idx,:], self.item_feature[idx,:]) # item_feature = (item_onehot, item_overview)
        return (features, self.labels[idx])

    def __len__(self):
        return self.count

class VanillaGMFDataset(Dataset):
    def __init__(self, inputs):
        self.user_feature = inputs[0][:,0].view(-1,1)
        self.item_feature = inputs[1][:,0].view(-1,1)
        self.labels = inputs[2].view(-1)

        self.count = inputs[0].shape[0]

    def __getitem__(self, idx):
        features = (self.user_feature[idx], self.item_feature[idx])
        return (features, self.labels[idx])

    def __len__(self):
        return self.count

class NeuMFDataset(Dataset):
    def __init__(self, inputs):
        self.user_feature_gmf = inputs[0][:,0].view(-1,1)
        self.item_feature_gmf = inputs[1][:,0].view(-1,1)

        self.user_feature_mlp = inputs[0]
        self.item_feature_mlp = inputs[1]

        self.labels = inputs[2].view(-1)

        self.count = inputs[0].shape[0]

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        features = (self.user_feature_gmf[idx], self.item_feature_gmf[idx], self.user_feature_mlp[idx,:], self.item_feature_mlp[idx,:])
        return (features, self.labels[idx])


