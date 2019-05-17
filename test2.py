"""
This file loads the data from the data directory and shows you how.
Feel free to change the contents of this file!
Do ensure these functions remain functional:
    - get_business(city, business_id)
    - get_reviews(city, business_id=None, user_id=None, n=10)
    - get_user(username)
"""
import numpy as np
import pandas as pd
import os
import json
import random
from collections import Counter
import math
import geopy.distance
import sklearn.metrics.pairwise as pw


DATA_DIR = "data"


def load_cities():
    """
    Finds all cities (all directory names) in ./data
    Returns a list of city names
    """
    return os.listdir(DATA_DIR)


def load(cities, data_filename):
    """
    Given a list of city names,
        for each city extract all data from ./data/<city>/<data_filename>.json
    Returns a dictionary of the form:
        {
            <city1>: [<entry1>, <entry2>, ...],
            <city2>: [<entry1>, <entry2>, ...],
            ...
        }
    """
    data = {}
    for city in cities:
        city_data = []
        with open(f"{DATA_DIR}/{city}/{data_filename}.json", "r") as f:
            for line in f:
                city_data.append(json.loads(line))
        data[city] = city_data
    return data


def get_business(city, business_id):
    """
    Given a city name and a business id, return that business's data.
    Returns a dictionary of the form:
        {
            name:str,
            business_id:str,
            stars:str,
            ...
        }
    """
    for business in BUSINESSES[city]:
        if business["business_id"] == business_id:
            return business
    raise IndexError(f"invalid business_id {business_id}")


def get_reviews(city, business_id=None, user_id=None, n=10):
    """
    Given a city name and optionally a business id and/or auser id,
    return n reviews for that business/user combo in that city.
    Returns a dictionary of the form:
        {
            text:str,
            stars:str,
            ...
        }
    """
    def should_keep(review):
        if business_id and review["business_id"] != business_id:
            return False
        if user_id and review["user_id"] != user_id:
            return False
        return True

    reviews = REVIEWS[city]
    reviews = [review for review in reviews if should_keep(review)]
    return random.sample(reviews, min(n, len(reviews)))


def get_user(username):
    """
    Get a user by its username
    Returns a dictionary of the form:
        {
            user_id:str,
            name:str,
            ...
        }
    """
    for city, users in USERS.items():
        for user in users:
            if user["name"] == username:
                return user
    raise IndexError(f"invalid username {username}")


CITIES = load_cities()
USERS = load(CITIES, "user")
BUSINESSES = load(CITIES, "business")
REVIEWS = load(CITIES, "review")
TIPS = load(CITIES, "tip")
CHECKINS = load(CITIES, "checkin")

def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    
    #make dataframe
    df_BUSINESS = pd.DataFrame()
    
    #make lists to append all values to
    all_ids = []
    all_names = []
    all_cities = []
    all_stars = []
    all_review_count = []
    all_is_open = []
    all_attributes = []
    all_categories = [] 
    all_latitude = []
    all_longitude = []
    all_attributes = []
    all_categories = []
    
    #search in the data and add values to the lists
    
    for city in BUSINESSES:
        for features in BUSINESSES[city]:
            all_ids.append(features['business_id'])
            all_names.append(features['name'])
            all_cities.append(features['city'])
            all_stars.append(features['stars'])
            all_review_count.append(features['review_count'])
            all_is_open.append(features['is_open'])
            all_latitude.append(features['latitude'])
            all_longitude.append(features['longitude'])
            all_categories.append(features['categories'])
            bag = []
            
            #check if the business has attributes and add a list of attributes
            if features['attributes'] != None:
                for element in features['attributes']:
                    if element:
                        bag.append(element)
            all_attributes.append(bag)

    #make columns in the dataframe 
    df_BUSINESS['busId'] = all_ids
    df_BUSINESS['name'] = all_names
    df_BUSINESS['city'] = all_cities
    df_BUSINESS['stars'] = all_stars
    df_BUSINESS['review_count'] = all_review_count
    df_BUSINESS['is_open'] = all_is_open
    df_BUSINESS['latitude'] = all_latitude
    df_BUSINESS['longitude'] = all_longitude
    df_BUSINESS['attributes'] = all_attributes
    df_BUSINESS['categories'] = all_categories
    
    if not city:
        city = random.choice(CITIES)
    return df_BUSINESS

df_BUSINESS = recommend(user_id=None, business_id=None, city=None, n=10)
df_BUSINESS.head()

def user_df():
    #create lists that represent columns
    all_bus_ids = []
    all_user_ids = []
    all_stars = []
    #add values from the data to lists
    for city in REVIEWS:
        for features in REVIEWS[city]: 
            if features['user_id'] not in all_user_ids:
                all_user_ids.append(features['user_id'])
                all_bus_ids.append(features['business_id'])
                all_stars.append(features['stars'])

            
    #create dataframe
    df_REVIEWS = pd.DataFrame()
    
    #add all the listvalues to the dataframe
    df_REVIEWS['userId'] = all_user_ids
    df_REVIEWS['stars'] = all_stars
    df_REVIEWS['busId'] = all_bus_ids
    
    return df_REVIEWS

df_USERS = user_df()
df_USERS.head()
#len(df_USERS)

def review_df():
    #create lists that represent columns
    all_bus_ids = []
    all_user_ids = []
    all_stars = []
    all_reviews = []
    #add values from the data to lists
    for city in REVIEWS:
        for features in REVIEWS[city]:
            all_bus_ids.append(features['business_id'])
            all_stars.append(features['stars'])
            all_user_ids.append(features['user_id'])
            all_reviews.append(features['text'])
            
    #create dataframe
    df_REVIEWS = pd.DataFrame()
    
    #add all the listvalues to the dataframe
    df_REVIEWS['userId'] = all_user_ids
    df_REVIEWS['stars'] = all_stars
    df_REVIEWS['busId'] = all_bus_ids
    df_REVIEWS['review'] = all_reviews
    
    return df_REVIEWS

df_REVIEWS = review_df()
df_REVIEWS.head()
#len(df_REVIEWS)


def reviews_test(reviews):
    #create lists that represent columns
    all_bus_ids = []
    all_user_ids = []
    all_stars = []
    #add values from the data to lists
    for city in REVIEWS:
        for features in REVIEWS[city]:
            all_user_ids.append(features['user_id'])
            
    review_amount = Counter(all_user_ids)
    
    all_active_users = []
    for user in review_amount:
        if review_amount[user] > 50:
            all_active_users.append(user)
 #   print(all_active_users)
    active_reviews_df = pd.DataFrame()
    for x in all_active_users:
        active_users = df_REVIEWS[(df_REVIEWS['userId'] == x)]
        active_reviews_df = active_reviews_df.append(active_users)
    
    return active_reviews_df

df_TESTREVIEWS = reviews_test(df_REVIEWS)
df_TESTREVIEWS.head()

def test_business():
    #create lists that represent columns
    all_business_ids = []
    all_user_ids = []
    all_stars = []
    #add values from the data to lists
    for city in REVIEWS:
        for features in REVIEWS[city]:
            all_business_ids.append(features['business_id'])
            
    review_amount = Counter(all_business_ids)
    #print(review_amount)
    all_active_bus = []
    active_bus_df = pd.DataFrame()
    for bus in review_amount:
        if review_amount[bus] > 300:
            all_active_bus.append(bus)
    for x in all_active_bus:
        active_bus = df_BUSINESS[(df_BUSINESS['busId'] == x)]
        active_bus_df = active_bus_df.append(active_bus)

    
    return active_bus_df

df_BUSTEST = test_business()
df_BUSTEST.head()

def test_users():
    active_users = set()
    
    for x in df_TESTREVIEWS['userId']:
        active_users.add(x)
    
    active_users = list(active_users)
    all_active_users = pd.DataFrame()
    all_active_users['userId'] = active_users
    
    return all_active_users

df_TESTUSERS = test_users()
df_TESTUSERS

def attribute_similarity(matrix, id1, id2):
    similar = 0
    bag = []
    
    #search for all features with the given id
    feature1 = matrix[(matrix['busId'] == id1)]['attributes'].item()
    feature2 = matrix[(matrix['busId'] == id2)]['attributes'].item()
    #append all the items to a bag of features of item 1
    for item1 in feature1:
        bag.append(item1)
        
    #append all the items to a bag of features of item 2
    for item2 in feature2:
        bag.append(item2)
        
    #counting all the words and see if the words in the bags are simalair
    count_bag = Counter(bag)
    total_words = len(bag)
    for element in count_bag:
        if count_bag[element] > 1:
            similar += count_bag[element]
    if total_words == 0:
        return total_words
    
    #return the percentage of similair attributes
    return similar/total_words

def categories_similarity(matrix, id1, id2):
    similar = 0
    bag = []
        
    #search for all features with the given id
    feature1 = matrix[(matrix['busId'] == id1)]['categories'].item()
    feature2 = matrix[(matrix['busId'] == id2)]['categories'].item()
    
    if feature1 == None or feature2 == None:
        return 0
        
    #append all the items to a bag of features of item 1
    for item1 in feature1:
        bag.append(item1)
        
    #append all the items to a bag of features of item 2
    for item2 in feature2:
        bag.append(item2)
        
    #counting all the words and see if the words in the bags are simalair
    count_bag = Counter(bag)
    total_words = len(bag)
    for element in count_bag:
        if count_bag[element] > 1:
            similar += count_bag[element]
    if total_words == 0:
        return total_words
        
    #return the percentage of similair categories
    return similar/total_words


def sim_matrix(matrix):
    #create an empty dataframe
    similarity_matrix = pd.DataFrame(matrix, index = matrix['busId'], columns = matrix['busId'])
    #get the index of the matrix
    business_ids = matrix['busId']
    
    #iterate over all the business ids and add the similarity in the matrix
    for business in business_ids:
        for business2 in business_ids:
            similarity_matrix.loc[business][business2] = ((attribute_similarity(matrix, business, business2)* 0.5) + categories_similarity(matrix, business, business2)) 
            if business2 == business:
                similarity_matrix.loc[business][business2] = 0

    return similarity_matrix

sim_matrix = sim_matrix(df_BUSTEST)
sim_matrix.head()

def split_data(data,d = 0.75):
    """ split data in a training and test set 
       `d` is the fraction of data in the training set"""
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]
training_set, test_set = split_data(df_TESTREVIEWS)
#training_set
test_set.head()

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

utility = utility_matrix(df_BUSTEST, df_TESTUSERS, df_TESTREVIEWS)
utility.head()


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

prediction = predict_ratings(sim_matrix, utility, test_set[['userId', 'busId', 'stars']])
prediction.head()

def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings
    
    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    predicted_ratings = predicted_ratings[predicted_ratings['predicted rating'] > 0]
    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()

mse(prediction)

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
mse_random(prediction)