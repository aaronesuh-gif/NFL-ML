#data
import pandas as pd #reading csv files
import numpy as np #math helper

#ml
from xgboost import XGBClassifier #model I use(?)
from sklearn.ensemble import RandomForestClassifier #model I use(?)
from sklearn.linear_model import LogisticRegression #control group basic model

#training
from sklearn.model_selection import train_test_split #splits data 
from sklearn.preprocessing import StandardScaler #normalization

#evaluation
from sklearn.metrics import (
    accuracy_score,  #accuracy %
    classification_report, #performance report
    confusion_matrix, #T/F, +/-
    roc_auc_score, #how well model distinguishes w/l
    log_loss) #confidence (lower = better)

#visualization
import matplotlib.pyplot as plt #creates plots/graphs
import seaborn as sns #makes better visuals

#saves models
import pickle




#reading csv file function

def loading_data():
    filepath = "data/spreadspoke_scores.csv"

    print('loading')
    df = pd.read_csv(filepath)

    #first few rows of data
    print('first few rows:')
    print(df.head())

    return df

#preparing data, clean for model
def preparing_data(df):
    print('preparing data')
    data = df.copy()

#creating target variable
# 1= home team wins, 0 = away team wins

    #checks if home team won using data, assigns True to 1 (home win) and False to 0
    data['home_wins'] = (data['score_home']>data['score_away']).astype(int)

    #removes games with missing values in either teams scores, future games or problem games
    data=data.dropna(subset=['score_home','score_away'])

    return data



#choose columns model will use
def select_features(data):
    feature_columns = [
        'spread_favorite',
        'over_under_line',
        'schedule_week',                 
    ]

    for i, col in enumerate(feature_columns):
        print(f"{i}: {col}")

    clean_data = data.dropna(subset = feature_columns)

    x = clean_data[feature_columns]
    y = clean_data['home_wins']

    dates = clean_data['schedule_date']

    return x, y, dates, clean_data

#split data into training and testing sets
def split_data(x,y,dates):

    'splitting by time, training on older games testing on newer games'

    print('splitting data')

    split_date = dates.quantile(0.8) #splitting at 80%, 80 for test

    print(f'training on games before: {split_date.date()}')
    print(f'testing on games after: {split_date.date()}')

    #create train and test sets
    training = dates < split_date
    testing = dates >= split_date

    x_training = x[training]
    x_testing = x[testing]
    y_training = y[training]
    y_testing = y[testing]


    print(f'training set size: {len(x_training)}')  #number of training set size 
    print(f'testing set size: {len(x_testing)}') #number of testing set size

    return x_training, x_testing, y_training, y_testing

#training model
def train_model(x_training, y_training):

#training xgboost model
    print('training model')
    
    model = XGBClassifier(
        n_estimators=125, # number of trees can change if too slow or bad
        learning_rate=0.1, #model learning rate
        max_depth=6, #depth of tree
        random_state=42, #reproducibility
        n_jobs=-1 #use cpu cores
    )

    model.fit(x_training, y_training)

    print ('model trained')

    return model