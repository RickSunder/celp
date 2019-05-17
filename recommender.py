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
import test

def recommend2(user_id=None, business_id=None, city=None, n=10):
    mat = get_matrix()
    top_recomm = all_recommendations(mat, user_id)
    return random.sample(top_recomm, n)

def recommend3(user_id=None, business_id=None, city=None, n=10):
    mat = get_matrix()
    top_recomm = all_recommend(mat, user_id, business_id)
    return random.sample(top_recomm, n)

def recommend(user_id=None, business_id=None, city=None, n=10):
    return home_logout()


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


def attribute_similarity(matrix, id1, id2):
    similar = 0
    bag = []
    feature1 = matrix[(matrix['busId'] == id1)]['attributes'].item()
    feature2 = matrix[(matrix['busId'] == id2)]['attributes'].item()
    
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
    feature1 = matrix[(matrix['busId'] == id1)]['categories'].item()
    feature2 = matrix[(matrix['busId'] == id2)]['categories'].item()
    
    for item1 in feature1 or []:
        bag.append(item1)
        
    for item2 in feature2 or []:
        bag.append(item2)
    
    if feature1 == None or feature2 == None:
        return 0
        
    count_bag = Counter(bag)
    total_words = len(bag)
    for element in count_bag:
        if count_bag[element] > 1:
            similar += count_bag[element]
    if total_words == 0:
        return total_words     
    return similar/total_words

def sim_matrix(matrix):
    similarity_matrix = pd.DataFrame(matrix, index = matrix['busId'], columns = matrix['busId'])
    business_ids = matrix['busId']
    for business in business_ids:
        for business2 in business_ids:
            similarity_matrix.loc[business][business2] = ((attribute_similarity(matrix, business, business2) * 0.5) + categories_similarity(matrix, business, business2)) 
            if business2 == business:
                similarity_matrix.loc[business][business2] = 0
    return similarity_matrix



def user_reviews(user_id, userdf):
    bus_ids = set()
    user_ids = userdf[(userdf['userId'] == user_id)]
    user_ids = user_ids[(user_ids['stars'] > 3)]
    for bus_id in user_ids['busId']:
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
    top_rec = []
    for i in recommendations:
        city = matrix[(matrix['busId'] == i)]['city'].item()
        city = city.lower()
        top_rec.append(get_business(city, i))
    return top_rec

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

def all_recommend(matrix, user_id, business_id):
    sim_mat = sim_matrix(matrix)
    recommendations = recommended_busids(sim_mat, business_id)
    top_rec = []
    for i in recommendations:
        city = matrix[(matrix['busId'] == i)]['city'].item()
        city = city.lower()
        top_rec.append(get_business(city, i))
    return top_rec

def home_logout():
    matrix = get_matrix()
    category = matrix['categories'].str.split(',').tolist()
    category_set = set()
    for x in category:
        for y in x:
            category_set.add(y)
    category_dict = dict()
    for item in category_set:
        temp = matrix.copy()
        check = temp[temp['categories'].str.contains(item)]
        category_dict[item] = check['busId'][check['stars'] >= 4.0].tolist()
        if category_dict[item] == []:
            category_dict.pop(item)
    temporary = set(category_dict.keys())
    randomnizer = random.sample(temporary, 15)
    rand = []
    random_business = []
    while len(rand) < 10:
        for cat in randomnizer:
            temp = random.sample(category_dict[cat], 1)
            if temp not in rand:
                rand.append(temp) 
    for busi_id in rand:
        for element in busi_id:
            city = matrix[(matrix['busId'] == element)]['city'].item()
            city = city.lower()
            random_business.append(get_business(city, element)) 
    return(random_business)

