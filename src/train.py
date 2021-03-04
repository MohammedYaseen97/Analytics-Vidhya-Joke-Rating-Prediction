import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

import config

def pred_matrix(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)                             #axis=1 means along the row
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    
    return pred

def run(fold, df):
    train_df = df[df["kfold"]!=fold].reset_index(drop=True)
    val_df = df[df["kfold"]==fold].reset_index(drop=True)
    
#    print(train_df.shape)

    n_users = train_df.user_id.unique().shape[0]
    n_jokes = train_df.joke_id.unique().shape[0]
    
    #Building a user_id x joke_id rating matrix
    rating_matrix = np.zeros((n_users, n_jokes))
    
    for line in train_df.itertuples():
#        print(line)
        rating_matrix[line[2]-1, line[3]-1] = line[4]
    
    user_similarity = pairwise_distances(rating_matrix, metric = 'cosine')
    item_similarity = pairwise_distances(rating_matrix.T, metric = 'cosine')
    
    matrix = pred_matrix(rating_matrix, item_similarity, type='item')
    
    #All the val results in one array
    pred_array = []
    
    for line in val_df.itertuples():
        pred_array.append(matrix[line[2]-1][line[3]-1])
    
    val_array = val_df["Rating"].values
    
    rmse = sqrt(mean_squared_error(val_array, pred_array))
    print("Fold: {}, RMSE : {}".format(fold, rmse))

if __name__ == "__main__":
    start_time = time.time()
    
    df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train_folds.csv'))
    
    run(0, df)
    run(1, df)
    run(2, df)
    run(3, df)
    run(4, df)
    
    end_time = time.time()
    print("Elapsed time: {}".format(end_time-start_time))