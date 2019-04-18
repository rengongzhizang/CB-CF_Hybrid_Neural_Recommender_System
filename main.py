import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from autoencoders import *
from recomDatasets import *
from utils import *
from preprocessData import *

## two kinds of embeddings: for GMF, for MLP
## user_features: user_tag, jobs, ages, genders, zipcode
## movie_features: genres, descriptions, (ratings?)
## user_id
## movie_id

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'data/'
    movie_data, movie_plot = item_features_loader(path)
    user_data, user_dict_size = user_features_loader(path)
    plot_word_to_ix, plot_max_length, plot_lines = make_words_dict(movie_plot)
    genre_word_to_ix, genre_max_length, genre_lines = make_genre_dict(movie_data)
    plot_onehot = make_onehot(plot_lines, plot_word_to_ix, plot_max_length)
    genre_onehot = make_onehot(genre_lines, genre_word_to_ix, genre_max_length)
    train_vec = [torch.load(path+'/user_features_train.pt'), \
        torch.load(path+'genre_features_train.pt'), torch.load(path+'labs_train.pt')]
    test_vec = [torch.load(path+'/user_features_test.pt'), \
        torch.load(path+'genre_features_test.pt'), torch.load(path+'labs_test.pt')]
    #pdb.set_trace()
    gmf(device, train_vec, test_vec, user_dict_size+1, len(genre_word_to_ix)+1 , len(plot_word_to_ix)+1)
    #mlp(device, train_vec, test_vec, user_dict_size+1, len(genre_word_to_ix)+1 , len(plot_word_to_ix)+1)
    #neumf(device, train_vec, test_vec, user_dict_size+1, len(genre_word_to_ix)+1 , len(plot_word_to_ix)+1)
    #pdb.set_trace()

if __name__ == "__main__":
    main()