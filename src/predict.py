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

train_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'test.csv'))

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

for line in test_df.itertuples():
    pred_array.append(matrix[line[2]-1][line[3]-1])

test_df["Rating"] = pred_array

test_df.to_csv(os.path.join(config.INPUT_PATH, "predictions_item_collab.csv"), columns = ["id", "Rating"], index = False)