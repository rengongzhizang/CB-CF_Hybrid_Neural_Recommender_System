import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from recomDatasets import *
from autoencoders import *
from NeuMF import *
import numpy as np
import pdb
from evaluation import *

def data_loader(path):
    df = pd.read_csv(path)
    headers = list(df)
    return headers, df

def tfidf_generator(corpus):  # return a scipy sparse tf-idf embedding, 
    tagsVecterizer = TfidfVectorizer(ngram_range=(1,2), min_df=1e-3, stop_words='english')                                              # preprocessing for autoencoder
    tagsVec = tagsVecterizer.fit_transform(corpus)
    return tagsVec.todense()
'''
    This function load the word_dict of plots' overview
'''
def make_words_dict(corpus): # corpus is a dictionary of strings
    word_to_ix = dict()
    max_length = 0
    lines = {}
    for i, line in corpus.items():
        line = line.lower()
        line = re.sub(r'[^\w\s]','',line)
        words = line.strip().split(" ")
        max_length = max(max_length, len(words))
        lines[i] = words
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix) + 1
    return word_to_ix, max_length, lines

'''
    This function load the word_dict of genres
'''
def make_genre_dict(corpus):
    word_to_ix = dict()
    max_length = 0
    lines = dict()
    for i, line in corpus.items():
        #line = line[1]
        #line = line.lower()
        #line[1].append(str(i))
        lines[i] = [str(i)] + line[1]
        max_length = max(len(lines[i]), max_length)
        for word in lines[i]:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix) + 1
    return word_to_ix, max_length, lines

def make_embed_vec(lines, word_to_ix, max_length): # lines is a list of lists of strings
    vec = torch.zeros((len(lines), max_length))
    for i,line in enumerate(lines):
        for j, word in enumerate(line):
            vec[i][j] = word_to_ix[word]
    return vec

def make_item_vec(ls, word_to_ix, max_length):
    vec = torch.zeros((1,max_length))
    for i, word in enumerate(ls):
        vec[0][i] = word_to_ix[word]
    return vec

def make_onehot(lines, word_to_ix, max_length):
    onehot = dict()
    for ids, words in lines.items():
        onehot_emb = [0] * max_length
        for i, word in enumerate(words):
            onehot_emb[i] = word_to_ix[word]
        onehot[ids] = onehot_emb
    return onehot

def list_to_vec(train_data, test_data):
    train_user_vec = torch.zeros((len(train_data),len(train_data[0][0])))
    train_genre_vec = torch.zeros((len(train_data), len(train_data[0][1][0])))
    train_plot_vec = torch.zeros((len(train_data), len(train_data[0][1][1])))
    train_labs = torch.zeros((len(train_data),1))
    test_user_vec = torch.zeros((len(test_data),len(test_data[0][0])))
    test_genre_vec = torch.zeros((len(test_data), len(test_data[0][1][0])))
    test_plot_vec = torch.zeros((len(test_data), len(test_data[0][1][1])))
    test_labs = torch.zeros((len(test_data),1))
    for i,j in enumerate(train_data):
        train_user_vec[i,:] = torch.tensor(j[0])
        train_genre_vec[i,:] = torch.tensor(j[1][0])
        train_plot_vec[i,:] = torch.tensor(j[1][1])
        train_labs[i,:] = torch.tensor(j[2])
    for i,j in enumerate(test_data):
        test_user_vec[i,:] = torch.tensor(j[0])
        test_genre_vec[i,:] = torch.tensor(j[1][0])
        test_plot_vec[i,:] = torch.tensor(j[1][1])
        test_labs[i,:] = torch.tensor(j[2])
    train_vec = [train_user_vec, [train_genre_vec, train_plot_vec], train_labs]
    test_vec = [test_user_vec, [test_genre_vec, test_plot_vec], test_labs]
    return train_vec, test_vec


def generator(device, net, loader):     # this net here is the encoder net
    embeddings = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        scores = net(inputs)            # batch_size * 50 
        embeddings.append(scores)
    embeddings = torch.cat(embeddings, 0)
    return embeddings

'''
def tag_encoder(device, tagsVec, em=30, es=50 ,dr=0.2): # users' tag autoencoder dim = 50, hidden_dim = 200, a sparse matrix
    tagNums, dim = tagsVec.shape
    trainVec, valVec = tagsVec[:tagNums - int(tagNums/4),:], tagsVec[tagNums - int(tagNums/4):,:]

    train_data = AutoEncoderDataset(trainVec)
    val_data = AutoEncoderDataset(valVec)
    data = AutoEncoderDataset(tagsVec)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)

    net = nn.Sequential(TagEncoder(dim, encoding_size=es, dropout_rate=dr), TagDecoder(dim, encoding_size=es, dropout_rate=dr))
    loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, train_loss_list, val_loss_list = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=em)
    embeds = generator(device, trained_net[0], loader)
    return embeds, train_loss_list, val_loss_list

def plot_encoder(device, plot_vec, word_to_ix, en=30, es=200):
    plot_num, dim = plot_vec.shape
    trainVec, valVec = plot_vec[:plot_num - int(plot_num/4),:], plot_vec[plot_num - int(plot_num/4):,:]
    
    train_data = OverviewDataset(trainVec)
    val_data = OverviewDataset(valVec)
    data = OverviewDataset(plot_vec)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)

    net = nn.Sequential(RNNEncoder(len(word_to_ix)+1, es), RNNDecoder(len(word_to_ix)+1, es))
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, train_loss_list, val_loss_list = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=en)
    embeds = generator(device, trained_net[0], loader)
    return embeds, train_loss_list, val_loss_list
'''

def train(device, net, loss_fn, optimizer, train_loader, test_vec, dataset, epoch_num, stacking=False):
    print('-'*10,'Start Training!','-'*10)
    best_params = 0.0
    best_hr = 0.0
    evaluations = []                                                                                #evaluations including hr and NDCG at each epoch
    for epoch in range(epoch_num):
        print('Epoch Num: {} / {} \n -------------------------'.format((epoch + 1), epoch_num))
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            #for i in range(len(inputs)):
                #inputs[i] = inputs[i].to(device)
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            if stacking:
                inputs[2] = inputs[2].to(device)
                inputs[3] = inputs[3].to(device)
            labels = labels.to(device)
            scores = net(inputs)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #pdb.set_trace()
            running_loss += loss.item() * inputs[0].size(0)
            running_acc += torch.sum(torch.tensor((torch.round(scores.view(-1)) == labels.data)))
            if i % 50 == 0:
                print('{}/{} iterations in this epoch has been trained'.format(i+1, len(train_loader)))    #iteration_nums
        epoch_loss = running_loss / len(train_loader)
        print('The loss of this epoch is {}'.format(epoch_loss))

        net.eval()
        hr, ndcg = hit_ratio(device, net, test_vec, dataset, stacking)
        evaluations.append([hr, ndcg])
        if hr > best_hr:
            best_hr = hr
            best_params = net.state_dict()
    net.load_state_dict(best_params)
    return net, evaluations

def save_net_evals(path, evaluations, net, name):
    file = open(path + '{}_evaluations.txt'.format(name), 'w')
    for x in evaluations:
        file.write('{} {}\n'.format(x[0],x[1]))
    torch.save(net.state_dict(), path + '{}_net.pt'.format(name))

def gmf(device, train_vec, test_vec, user_dict_size, genre_dict_size, plot_dict_size, epoch_num=50, embed_dim=32):
    #pdb.set_trace()
    train_data = VanillaGMFDataset(train_vec)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4000, shuffle=True)

    net = VanillaGMF(user_dict_size, genre_dict_size, embed_dim).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, evaluations = train(device, net, loss_fn, optimizer, train_loader, test_vec, VanillaGMFDataset, epoch_num)
    save_net_evals('data/', evaluations, trained_net, 'GMF')


def mlp(device, train_vec, test_vec, user_dict_size, genre_dict_size, plot_dict_size, epoch_num=10, embed_dim=512):
    #pdb.set_trace()
    train_data = RecsysDataset(train_vec)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5000, shuffle=True)

    net = MLP(user_dict_size, genre_dict_size, plot_dict_size, embed_dim).to(device)
    #pdb.set_trace()
    loss_fn = nn.BCELoss()
    optimizer = optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, evaluations = train(device, net, loss_fn, optimizer, train_loader, test_vec, RecsysDataset, epoch_num)
    save_net_evals('data/', evaluations, trained_net, 'MLP')

def neumf(device, train_vec, test_vec, user_dict_size, genre_dict_size, plot_dict_size, epoch_num=10, embed_dim=512):
    #pdb.set_trace()
    train_data = NeuMFDataset(train_vec)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5000, shuffle=True)
    gmf_net = VanillaGMF(user_dict_size, genre_dict_size, 32).to(device)
    mlp_net = MLP(user_dict_size, genre_dict_size, plot_dict_size, embed_dim).to(device)
    if device == torch.device('cuda'):
        gmf_net.load_state_dict(torch.load('data/GMF_net.pt'))
        mlp_net.load_state_dict(torch.load('data/MLP_net.pt'))
        gmf_net.to(device)
        mlp_net.to(device)
    else:
        gmf_net.load_state_dict(torch.load('data/GMF_net.pt', map_location='cpu'))
        mlp_net.load_state_dict(torch.load('data/MLP_net.pt', map_location='cpu'))
    for param in gmf_net.parameters():
        param.requires_grad = False
    for param in mlp_net.parameters():
        param.requires_grad = False
    net = NeuMF(gmf_net, mlp_net).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained_net, evaluations = train(device, net, loss_fn, optimizer, train_loader, test_vec, NeuMFDataset, epoch_num, stacking=True)
    save_net_evals('data/', evaluations, trained_net, 'NeuMF')