import pickle
import numpy as np
#from bettingcalc import ... whatever i name the betting calculator methods

def load_model():
    'load trained model'
    
    model_path = '../dev/model/nfl_model.pkl'
    
    print('loading model')



    try: 
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print('model loaded')
            return model
        
    except FileNotFoundError:
        print('model not found')
        return None
    




def get_user_input():
#asks user for game info
    home_team = input('Enter home team: ')
    home_team = home_team.strip
    if home_team.lower() == 'quit':
        return None
    
    away_team = input('Enter away team: ')
    away_team = away_team.strip
    if away_team.lower() == 'quit':
        return None
    
    #get betting info
    
