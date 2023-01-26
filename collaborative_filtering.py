
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        self.user_item_matrix = matrix

        # User - User based collaborative filtering
        if self.strategy == 'user':
            # each row in the df represents a vector of a user
            start_time = time.time()
            # compute mean
            user_rating_avg = np.nanmean(self.user_item_matrix, axis=1).reshape(-1, 1)
            # adding very small number (0.001) to avoid division by zero
            rating_difference = (self.user_item_matrix - user_rating_avg) + 0.001
            # nan values are replaced with 0 value
            rating_difference[np.isnan(rating_difference)] = 0
            # compute user similarity using the cosine function
            self.similarity = 1 - pairwise_distances(rating_difference, metric='cosine')
            # compute the dot product of the rating_difference and normalize it
            # then add to that the 'user_rating_avg'
            predictions = user_rating_avg + self.similarity.dot(rating_difference) / np.array(
                [np.abs(self.similarity).sum(axis=1)]).T
            # store the prediction matrix in 'pred' dataframe
            self.pred = pd.DataFrame(predictions, index=matrix.index, columns=matrix.columns)
            # round up to 2 numbers
            self.pred = self.pred.round(2)

            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        # Item - Item based collaborative filtering
        elif self.strategy == 'item':
            # each column in the df represents a vector of an item
            start_time = time.time()
            # transpose the df so now the rows are the items
            item_user_matrix = self.user_item_matrix.to_numpy()
            # compute mean
            item_rating_avg = np.nanmean(item_user_matrix, axis=1).reshape(-1, 1)
            # adding very small number (0.001) to avoid division by zero
            rating_difference = (item_user_matrix - item_rating_avg) + 0.001
            # nan values are replaced with 0 value
            rating_difference[np.isnan(rating_difference)] = 0
            # compute user similarity using the cosine function
            self.similarity = 1 - pairwise_distances(rating_difference.T, metric='cosine')
            # compute the dot product of the rating_difference and normalize it
            # then add to that the 'item_rating_avg'
            predictions = item_rating_avg + rating_difference.dot(self.similarity) / np.array(
                [np.abs(self.similarity).sum(axis=1)])
            # store the prediction matrix in 'pred' dataframe
            self.pred = pd.DataFrame(predictions, index=matrix.index, columns=matrix.columns)
            # round up to 2 numbers
            self.pred = self.pred.round(2)

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):
        if self.strategy == 'user':
            # try to find user_id in data
            try:
                row = self.user_item_matrix.index.get_loc(user_id)
            except KeyError:
                return None
            return self.get_top_k(row, k)

        elif self.strategy == 'item':
            try:
                row = self.user_item_matrix.index.get_loc(user_id)
            except KeyError:
                return None
            return self.get_top_k(row, k)

    def get_top_k(self, row, k):
        # save the user row before nan values are removed from it
        user_row_with_nan = self.user_item_matrix.iloc[row]
        # now save the user row from pred
        user_row_from_pred = self.pred.iloc[row]
        # predict items the user didn't rate
        user_row_unrated = user_row_from_pred[user_row_with_nan.isna()]
        top_k_products = []
        for _ in range(k):
            max_idx = user_row_unrated.idxmax()
            # assign current max with 0, so it will not be max again
            user_row_unrated[max_idx] = 0
            top_k_products.append(max_idx)
        return top_k_products

