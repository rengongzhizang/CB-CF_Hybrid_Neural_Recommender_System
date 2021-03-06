# CB-CF_Hybrid_Neural_Recommender_System
This is a PyTorch implementation of a  Hybrid Recommendation System Based on Neural Networks inspired by He's paper [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). This is a course project of EECS 598-012 data-mining in Winter 2019 made by Yuanhang Cui, Weijie Sun, and Shiyu Wang.

Here is CB-CF Hybrid Neural Recommender System:

<p align="center">
<img width="460" height="300" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/model.png">
</p>

You can see the pdf file for more details.

Our group:
![Team](https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/team.jpg)

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
<br />
<br />
<br />
<br />

We have pre-combined all date you need into the folder data.

All you need to do are: `python preprocessData.py`

Several PyTorch Tensor such as `genre_features_train.pt`, `genre_features_test.pt`, `user_features_train.pt`, `user_features_test.pt`, `plot_features_train.pt`, and `plot_features_test.pt` would be generated and saved in the folder of data.

## Training
Run `python main.py`

Updates on April 22nd, 2019: a shell script - MAKEFILE.sh - has been added.

Run `chmod +x ./MAKEFILE.sh`
Then run `./MAKEFILE.sh`

## Our performance

### Experiment 1: Hyper-parameter Tuning

<p>
<img width="270" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/Picture1.png" align="left">
<img width="270" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/Picture2.png" align="left">
<img width="270" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/Picture3.png" align="left">
</p> <br /><br /><br /><br /><br /><br />
<br /><br /><br /><br /><br />

We could figure out that the best `drop_out_rate = 0.3`, best number of layers of MLP is `layers = 5`, and best layers' dimensions are `[32, 16 ,8 ,4, 2]`

### Experiment 2: Pretrained vs. No Pretrained

<p align="center">
<img width="820" src="https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/Picture4.png">
</p><br />

Our model out-performed He's CF only Neural RS with performance shown as follows:

![Performance](https://github.com/rengongzhizang/CB-CF_Hybrid_Neural_Recommender_System/blob/master/images/performance.png)

