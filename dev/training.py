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

    