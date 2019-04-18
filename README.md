# CB-CF_Hybrid_Neural_Recommender_System
This is a PyTorch implementation of a  Hybrid Recommendation System Based on Neural Networks inspired by He's paper [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). This is a course project of EECS 598-012 data-mining in Winter 2019 made by Yuanhang Cui, Weijie Sun, and Shiyu Wang.

Here is CB-CF Hybrid Neural Recommender System:

<p align="center">
<img width="460" height="300" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/model.png">
</p>

You can see the pdf file for more details.

## Dependencies
* python 3.6

* PyTorch
![PyTorch](https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/pytorch.png)

please `pip install` the following modules into the project folder:
* numpy
* scipy
* Pandas


## Data preparation
We used the dataset of [MovieLens](https://grouplens.org/datasets/movielens/1m/) of the version of ml-1m,  and the dataset of The [Movie Database(TMDb)](https://www.themoviedb.org).

Files in ml-1m of Movielens contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users; We also request and parse data such like movie plot overview from TMDb via open API as a complement to the dataset of MovieLens. 

<p>
<img width="300" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/movielens.png" align="left">
<img width="100" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/tmdb.png" align="left">
</p>


We have pre-combined all date you need into the folder data.

All you need to do are: `python preprocessData.py`

Several PyTorch Tensor such as `genre_features_train.pt`, `genre_features_test.pt`, `user_features_train.pt`, `user_features_test.pt`, `plot_features_train.pt`, and `plot_features_test.pt` would be generated and saved in the folder of data.
