import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TagEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=200, encoding_size=50, dropout_rate=0.2):
        super(TagEncoder, self).__init__()
        self.tagEncoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, encoding_size),
            nn.BatchNorm1d(encoding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):      # prepocessed TF-IDF embeddings with dimension: 
        x = self.tagEncoder(x)
        return x

class TagDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=200, encoding_size=50, dropout_rate=0.2):
        super(TagDecoder, self).__init__()
        self.tagDecoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, vocab_size),
            nn.BatchNorm1d(vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.tagDecoder(x)
        return x
'''
    This is an encoder designed to embed plot overviews
'''
class RNNEncoder(nn.Module):
    def __init__(self, dict_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    def forward(self, input, hidden=None):
        embeded = self.embedding(input.long())#.view(len(input),1,-1) # batch * length * emb_dim
        output, hidden = self.gru(embeded, hidden)
        return output, hidden

'''
class RNNDecoder(nn.Module):
    def __init__(self, dict_size, hidden_size):
        super(RNNDecoder, self).__init__()
        self.dict_size = dict_size
        self.hidden_size = hidden_size
        #self.batch_size = batch_size
        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, dict_size)
    def forward(self, input, hidden):
        embeded = self.embedding(input.long()).view(len(input),1,-1)
        temp, hidden = self.gru(embeded, hidden)
        output = temp.view(len(input),-1)
        #temp = self.linear(flatten)
        output = F.log_softmax(self.linear(output), dim=1)
        return output
'''

'''
    Those are encoder designed to embed user/item (except overview) features
'''
class UserEncoder(nn.Module):
    def __init__(self, dict_size, embed_size):
        super(UserEncoder, self).__init__()
        self.embedding = nn.Embedding(dict_size, embed_size)            # batch * length * emb_dim
    
    def forward(self, x):                                               # x = user_onehot
        user_embed = self.embedding(x).mean(dim=1)                      # batch * emb_dim
        return user_embed

class ItemEncoder(nn.Module):
    def __init__(self, dict_size, embed_size):
        super(ItemEncoder, self).__init__()
        self.embedding = nn.Embedding(dict_size, embed_size)

    def forward(self, x):                                               # item_features = (item_onehot, item_overview), x = item_features[0]
        item_embed = self.embedding(x).mean(dim=1)
        return item_embed