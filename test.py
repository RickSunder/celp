from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from collections import Counter
import random
import pandas as pd
from pprint import pprint
from collections import Counter
import random
import pandas as pd
from collections import Counter
import data
import recommender
import numpy as np

def split_data(data,d = 0.75):
    """ split data in a training and test set 
       `d` is the fraction of data in the training set"""
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]


def utility_matrix(bus_df, user_df, review_df):
    #create an empty dataframe
    utility_matrix = pd.DataFrame(index = bus_df['busId'], columns = user_df['userId'])
    rating_amount = []
    #get the index of the dataframes
    business_ids = bus_df['busId']
    user_ids = user_df['userId']
    
    #iterate over all the business ids and add the values if possible
    for business in business_ids:
        for user in user_ids:
            rating = review_df[(review_df['busId'] == business) & (review_df['userId'] == user)]['stars']
            if len(rating) is 1:
                rating_amount.append(rating)
                utility_matrix.loc[business][user] = rating.item()
    
    print(len(rating_amount))
    return utility_matrix


def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return 0
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return 0

def predict_ratings(similarity, utility, to_predict):
    """Predicts the predicted rating for the input test data.
    
    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each movie (rows). 
                  If a user did not rate an item the value np.nan is assumed. 
    to_predict -- A dataFrame containing at least the columns movieId and userId for which to do the predictions
    """
    #copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    #apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['userId'], row['busId']), axis=1)
    return ratings_test_c

def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    predicted_ratings = predicted_ratings[predicted_ratings['predicted rating'] > 0]
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()


def mse_random(prediction):
    random_list = []
    df_copy = prediction.copy()

    #iterate over the series and add values between 0.5 and 5 with an interval of 0.5  
    for x in df_copy.values:
        random_list.append(np.random.choice(np.arange(0.5 , 5.5 , 0.5)))

    #add listvalues to series
    df_copy['predicted rating'] = random_list

    #calculate the mean squared error 
    mse_random = mse(df_copy) 
    
    return mse_random
