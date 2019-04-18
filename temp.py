import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import omdb
import pdb
from utils import *
from autoencoders import *
from preprocessData import *


##path = "data/meta_data.csv"
##df = pd.read_csv(path)
#print(data_loader(path))
##corpus = df['user_tag']
##tagsVecterizer = TfidfVectorizer(ngram_range=(1,2), min_df=1e-3, stop_words='english')
##tagsVec = tagsVecterizer.fit_transform(corpus)
#print(tagsVecterizer.get_feature_names())
##print(tagsVec.todense().shape)
#print(type(tagsVec.todense()))
#print(torch.from_numpy(tagsVec.todense()[0]).type(torch.FloatTensor))
#tagsVec = tfidf_generator(corpus)
#print(len(tagsVec))
#tagNums, dim = tagsVec.shape
#print(dim)
#print(int(tagNums/3))
#trainVec, valVec = tagsVec[:tagNums - int(tagNums/3),:], tagsVec[tagNums - int(tagNums/3):,:]
#print(trainVec.shape, valVec.shape)
#net = nn.Sequential(TagEncoder(1000),TagDecoder(1000))
#print(net[0])
#df = pd.read_csv("data/movies.csv")
#genres = df['genres']
#genresBag, max_length, lines = make_words_dict(genres)
#genresVec = make_embed_vec(lines, genresBag, max_length)
#print(genresVec[:5,:])
'''
path = "data/ratings.dat"
all_ratings, user_nums = ratings_loader(path)
print(list(all_ratings)[-1], user_nums)
'''
'''
path = "data/movies.csv"
_, df = data_loader(path)
nanlist = df['overview'].index[df['overview'].isna()]
'''
'''
print(type(df['1'][nanlist[0]]))
#client = omdb.OMDBClient(apikey='3135db50')

omdb.set_default('apikey', '3135db50')
print(omdb.title('League of Their Own', year=1992)['plot'])

for i in nanlist:
    print(df['1'][i])
'''
'''
nandict = []
omdb.set_default('apikey', '3135db50')
temp = df['1'][126]
if max(temp.find(','), temp[:-7].find('(')) == -1:
    with omdb.title(temp[0:-7], year=int(temp[-5:-1])) as plot:
        nandict.append(plot)
print(nandict)
'''
'''
nandict = []
for i in nanlist:
    temp = df['1'][i]
    omdb.set_default('apikey', '3135db50')
    if max(temp.find(','), temp[:-7].find('(')) == -1:
        plot = omdb.title(temp[0:-7], year=int(temp[-5:-1]))
        nandict.append(plot['plot']) if bool(plot) else nandict.append(None)
    else:
        if temp[:-7].find('(') == -1:
            plot = omdb.title(temp[0:temp.find(',')], year=int(temp[-5:-1]))
            nandict.append(plot['plot']) if bool(plot) else nandict.append(None)
        else:
            try:
                plot = omdb.title(temp[temp[:-7].find('('):temp[:-7].find(')')], year=int(temp[-5:-1]))
            except:
                nandict.append(None)
                continue
            if not plot:
                plot = omdb.title(temp[0:temp[:-7].find(',')], year=int(temp[-5:-1]))
                nandict.append(plot['plot']) if bool(plot) else nandict.append(None)
            else:
                nandict.append(plot['plot'])

print(len(nandict))
with open('data/temp_overview.csv','w') as f:
    for plot in nandict:
        f.write('%s' % plot)
        f.write('\n')
    f.close()
'''
'''
with open('data/temp_overview.csv', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line != 'N/A' or line != 'None':
            df['overview'][nanlist[i]] = line
print(df['overview'].isna().sum())
df.to_csv('data/overview_modified.csv')
'''
path = 'data/'
movie_data, movie_plot = item_features_loader(path)
#print(movie_data, movie_plot)
ratings, rating_num = ratings_loader(path + 'ratings.dat')
plotlist = list(movie_plot.values())
pdb.set_trace()