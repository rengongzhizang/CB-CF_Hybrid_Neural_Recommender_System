import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from autoencoders import *

class GMF(nn.Module):
    def __init__(self, user_dict_size, genre_dict_size, plot_dict_size, emb_dim, dropout_rate=0.1):            
        super(GMF, self).__init__()
        self.embedding_user = nn.Embedding(user_dict_size, emb_dim)
        self.embedding_genre = nn.Embedding(genre_dict_size, emb_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        user_emb = self.batchnorm(self.dropout(self.embedding_user(x[0].long()).mean(dim=1)))                 #after embeddings: batch * feature_num * (2)emb_dim
        genre_emb = self.batchnorm(self.dropout(self.embedding_genre(x[1].long()).mean(dim=1)))               #after pooling: batch * (2)emb_dim
        return self.sigmoid((user_emb * genre_emb).sum(dim=1))

class MLP(nn.Module):
    def __init__(self, user_dict_size, genre_dict_size, embed_dim, hidden_1=128, hidden_2=64, hidden_3=32, hidden_4=8, hidden_5=1, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.embedding_user = nn.Embedding(user_dict_size, embed_dim)
        self.embedding_genre = nn.Embedding(genre_dict_size, embed_dim)
        self.feature_num = embed_dim * 2
        self.linear_1 = nn.Linear(self.feature_num, hidden_1)
        self.linear_2 = nn.Linear(hidden_1, hidden_2)
        self.linear_3 = nn.Linear(hidden_2, hidden_3)
        self.linear_4 = nn.Linear(hidden_3, hidden_4)
        self.linear_5 = nn.Linear(hidden_4, hidden_5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(self.feature_num)

    def forward(self, x):
        user_emb = self.embedding_user(x[0].long()).mean(dim=1)                                #after embeddings: batch * feature_num * emb_dim
        genre_emb = self.embedding_genre(x[1].long()).mean(dim=1)                              #after pooling: batch * emb_dim
        features = self.batchnorm(torch.cat([user_emb ,genre_emb], dim=1))
        layer_1 = self.dropout(self.relu(self.linear_1(features)))
        layer_2 = self.relu(self.linear_2(layer_1))
        layer_3 = self.dropout(self.relu(self.linear_3(layer_2)))
        output = self.sigmoid(self.linear_5(self.relu(self.linear_4(layer_3))))
        return output

class NeuMF(nn.Module):
    def __init__(self, Model_GMF, Model_MLP):
        super(NeuMF, self).__init__()
        self.model_gmf = Model_GMF
        self.model_mlp = Model_MLP
        self.classifier = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gmf = self.model_gmf((x[0], x[1])).squeeze(dim=2)
        mlp = self.model_mlp((x[2], x[3]))
        cat = torch.cat([gmf, mlp], dim=1)
        return self.sigmoid(self.classifier(cat))


class VanillaGMF(nn.Module):
    def __init__(self, user_dict_size, genre_dict_size, emb_dim, dropout_rate=0.1):            
        super(VanillaGMF, self).__init__()
        self.embedding_user = nn.Embedding(user_dict_size, emb_dim)
        self.embedding_genre = nn.Embedding(genre_dict_size, emb_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(emb_dim)
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, x):
        user_emb = self.embedding_user(x[0].long())               #after embeddings: batch * feature_num * emb_dim
        genre_emb = self.embedding_genre(x[1].long())              #after pooling: batch * (2)emb_dim
        return self.sigmoid(self.linear(torch.mul(user_emb, genre_emb)))