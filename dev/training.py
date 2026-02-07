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

