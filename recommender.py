from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from collections import Counter
import random
import pandas as pd
from pprint import pprint
from collections import Counter
import random
import pandas as pd
from collections import Counter


def recommend2(user_id=None, business_id=None, city=None, n=10):
    mat = get_matrix()
    top_recomm = all_recommendations(mat, user_id)
    return random.sample(top_recomm, n)

def recommend(user_id=None, business_id=None, city=None, n=10):
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)	


def get_matrix():
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
    df_BUSINESS = pd.DataFrame()
    
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
            if features['attributes'] != None:
                for element in features['attributes']:
                    if element:
                        bag.append(element)
            all_attributes.append(bag)


    df_BUSINESS = df_BUSINESS.reindex(all_ids)
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

def attribute_similarity(matrix, id1, id2):
    similar = 0
    bag = []
    feature1 = matrix[(matrix.index == id1)]['attributes'].item()
    feature2 = matrix[(matrix.index == id2)]['attributes'].item()
    
    for item1 in feature1:
        bag.append(item1)
        
    for item2 in feature2:
        bag.append(item2)
        
    count_bag = Counter(bag)
    total_words = len(bag)
    for element in count_bag:
        if count_bag[element] > 1:
            similar += count_bag[element]
    if total_words == 0:
        return total_words       
    return similar/total_words

def categories_similarity(matrix, id1, id2):
    similar = 0
    bag = []
    feature1 = matrix[(matrix.index == id1)]['categories'].item()
    feature2 = matrix[(matrix.index == id2)]['categories'].item()
    
    for item1 in feature1 or []:
        bag.append(item1)
        
    for item2 in feature2 or []:
        bag.append(item2)
        
    count_bag = Counter(bag)
    total_words = len(bag)
    for element in count_bag:
        if count_bag[element] > 1:
            similar += count_bag[element]
    if total_words == 0:
        return total_words     
    return similar/total_words

def sim_matrix(matrix):
    similarity_matrix = pd.DataFrame(matrix, index = matrix.index, columns = matrix.index)
    business_ids = matrix.index
    for business in business_ids:
        for business2 in business_ids:
            similarity_matrix.loc[business][business2] = (attribute_similarity(matrix, business, business2) * categories_similarity(matrix, business, business2)) 
            if business2 == business:
                similarity_matrix.loc[business][business2] = 0
    return similarity_matrix

def user_df():
    all_bus_ids = []
    all_user_ids = []
    all_stars = []
    all_reviews = []
    for city in REVIEWS:
        for features in REVIEWS[city]:
            all_bus_ids.append(features['business_id'])
            all_stars.append(features['stars'])
            all_user_ids.append(features['user_id'])
            all_reviews.append(features['text'])
               
    df_REVIEWS = pd.DataFrame()
    df_REVIEWS = df_REVIEWS.reindex(all_user_ids)
    df_REVIEWS['stars'] = all_stars
    df_REVIEWS['business_id'] = all_bus_ids
    df_REVIEWS['review'] = all_reviews   
    return df_REVIEWS

def user_reviews(user_id, userdf):
    bus_ids = set()
    user_ids = userdf[(userdf.index == user_id)]
    for bus_id in user_ids['business_id']:
        bus_ids.add(bus_id)
    return bus_ids


def recommended_busids(sim_matrix, business_id):   
    best_sim = sim_matrix.nlargest(10, business_id)
    return list(best_sim[business_id].index)

def all_recommendations(matrix, user_id):
    sim_mat = sim_matrix(matrix)
    userdf = user_df()
    review_list = user_reviews(user_id, userdf)
    recommendations = []
    for business_id in review_list:
        rec_bus = recommended_busids(sim_mat, business_id)
        for bus_id in rec_bus:
            if bus_id not in recommendations:
                recommendations.append(bus_id)
    pprint(recommendations)
    return recommendations

