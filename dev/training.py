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


    #converting schedule date to datetime so can be split properly 
    data['schedule_date'] = pd.to_datetime(data['schedule_date'])

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


    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')

    clean_data = data.dropna(subset = feature_columns + ['home_wins'])

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

#evaluate, get metrics and accuracy
def evaluate_model(model, x_training , y_training, x_testing, y_testing):

#testing model performance

    print('model testing')

    #training set performance
    y_train_pred = model.predict(x_training)
    train_accuracy = accuracy_score(y_training, y_train_pred)

    #testing set performance
    y_test_pred = model.predict(x_testing)
    test_accuracy = accuracy_score(y_testing, y_test_pred)

    #probability prediction
    #gives confidence scores 0-1
    y_test_probability = model.predict_proba(x_testing)[:,1] #probability of home win

    #additional metrics
    auc = roc_auc_score(y_testing, y_test_probability) #how well model distinguishes w/l
    logloss = log_loss(y_testing, y_test_probability) #confidence (lower = better)


    #printing results

    print('accuracy:')
    print( f"training:{train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print( f"testing: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print("other metrics:")
    print(f"AUC: {auc:.4f} (highers = better, 0.5 = random, 1= perf)")
    print(f"Log Loss: {logloss:.4f} (lowers = better)")


    #confusion matrix
    print('confusion matrix:')
    conf_matrix = confusion_matrix(y_testing, y_test_pred)
    print(conf_matrix)

    print('reading the matrix:')
    print('correctly predicted away team wins:', conf_matrix[0,0])
    print('incorrectly predicted home team wins:', conf_matrix[0,1])
    print('incorrectly predicted away team wins:', conf_matrix[1,0])
    print('correctly predicted home team wins:', conf_matrix[1,1])

    #feature importance
    print('feature importance:')
    feature_importances = model.feature_importances_
    features = x_training.columns

    for feature, importance in sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")


    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'auc': auc,
        'logloss': logloss,
        'confusion_matrix': conf_matrix
    }


#save model to file

def save_model(model, filepath = '../dev/model/nfl_model.pkl'):
   
   #saving model to file so dont have to train each time
    print(f'saving model to {filepath}')
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True) #creates directory if doesnt exists, if exists just continues
    

    #save the model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f) #saves model to file in binary format through pickle ! yay !


    print('model saved')
    print('to load model later, use:')
    print('with open("path_to_model.pkl", "rb") as f:')
    print('    model = pickle.load(f)')


#main function runs all steps in correct order

def main():
    'runs entire training pipeline'

    print('nfl game prediction - training pipeline')

    #load data
    df = loading_data()

    # prepare data
    data = preparing_data(df)

    #select features and target
    x, y, dates, clean_data = select_features(data)

    #split data into training and testing sets
    x_training, x_testing, y_training, y_testing = split_data(x,y,dates)

    #train
    model = train_model(x_training, y_training)

    #evaluate
    results = evaluate_model(model, x_training , y_training, x_testing, y_testing)

    #save model
    save_model(model)

    print('training pipeline complete')
    print('results:')

    return model, results



if __name__ == "__main__":
    model, results = main() #runs when executing :python training.py


    