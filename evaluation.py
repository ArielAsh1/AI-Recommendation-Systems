# Ariel Ashkenazy 208465096

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(test_set, cf):
    # all rating predictions from cf.pred
    predictions = []
    # all mean ratings for each user_id
    benchmark_predictions = []
    mean_dataframe = cf.user_item_matrix.mean(axis=1)
    for row in test_set.itertuples():
        prediction = cf.pred.loc[row.UserId, row.ProductId]
        predictions.append(prediction)
        benchmark_prediction = mean_dataframe.loc[row.UserId]
        benchmark_predictions.append(benchmark_prediction)

    actuals = np.array(test_set.Rating)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    benchmark = benchmark_predictions
    benchmark_mse = mean_squared_error(actuals, benchmark)
    benchmark_rmse = np.sqrt(benchmark_mse)
    print("RMSE for {}-based CF: {}".format(cf.strategy, rmse.round(5)))
    print("benchmark RMSE: {}".format(benchmark_rmse.round(5)))


def precision_at_k(test_set, cf, k):
    precisions = []
    benchmark_precisions = []
    # get the avg rating of each item
    avg_ratings = cf.user_item_matrix.mean()
    # hold the k indices with the largest avg_ratings
    benchmark_recommended_items = avg_ratings.nlargest(k).index.tolist()

    for user_id in test_set.UserId.unique():
        recommended_items = cf.recommend_items(user_id, k)
        # compute items for user_id with rating 3+, and get the product id using Pandas .query
        relevant_items = list(test_set.query("UserId == @user_id and Rating >= 3").ProductId)
        # intersection between recommended_items and relevant_items
        relevant_recommended_items = set(recommended_items).intersection(set(relevant_items))
        # intersection between benchmark_recommended_items and relevant_items
        benchmark_relevant_recommended = set(benchmark_recommended_items).intersection(set(relevant_items))

        if relevant_items and recommended_items:
            # calculate precision for each user
            precisions.append(len(relevant_recommended_items) / len(recommended_items))
            benchmark_precisions.append(len(benchmark_relevant_recommended) / len(recommended_items))

    # average off all precisions
    test_precision = sum(precisions) / len(precisions)
    print("Precision@{} user-based CF (test set): {}".format(k, round(test_precision, 5)))
    benchmark_precision = sum(benchmark_precisions) / len(benchmark_precisions)
    print("Precision@{} highest-ranked Benchmark: {}".format(k, round(benchmark_precision, 5)))


def recall_at_k(test_set, cf, k):
    recalls = []
    benchmark_recalls = []
    avg_ratings = cf.user_item_matrix.mean()
    # holds the k largest avg_ratings
    benchmark_recommended_items = avg_ratings.nlargest(k).index.tolist()

    for user_id in test_set.UserId.unique():
        recommended_items = cf.recommend_items(user_id, k)
        # compute items for user_id with rating 3+, and get the product id using Pandas .query
        relevant_items = list(test_set.query("UserId == @user_id and Rating >= 3").ProductId)
        # intersection between recommended_items and relevant_items
        relevant_items_recommended = set(recommended_items).intersection(set(relevant_items))
        # intersection between benchmark_recommended_items and relevant_items
        benchmark_relevant_recommended = set(benchmark_recommended_items).intersection(set(relevant_items))

        if relevant_items and recommended_items:
            # compute precision for each user
            recalls.append(len(relevant_items_recommended) / len(relevant_items))
            benchmark_recalls.append(len(benchmark_relevant_recommended) / len(relevant_items))

    # average off all precisions
    test_recall = sum(recalls) / len(recalls)
    print("Recall@{} user-based CF (test set): {}".format(k, round(test_recall, 5)))
    benchmark_recall = sum(benchmark_recalls) / len(benchmark_recalls)
    print("Recall@{} highest-ranked Benchmark: {}".format(k, round(benchmark_recall, 5)))

