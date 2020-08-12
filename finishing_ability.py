# -*- coding: utf-8 -*-
"""

@author: charles william
"""

# A script to produce a finishing ability metric based on the *concept* of post-shot xG based on the phyiscal properties of
# a shot once taken

# see https://www.opengoalapp.com/finishing-ability for full write-up of the method


import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

from GetMatchDates import GetMatchDates
from PrepareIO import PrepareIO


from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
import xgboost as xgb
import matplotlib.pyplot as plt



# load in StatsBomb shot data from a snapshot file - you can also load in from the API as well with split=True
# then just combine the shot dataframes from each match into a single frame afterwards e.g:   
    
#-------------
#grouped_event_list = []
#for match_id in match_ids:
    #grouped_events = sb.events(match_id=match_id, split=True)
    #grouped_event_list.append(grouped_events)


#select the event group required and a list of dataframes will be returned
#shots_frames = [i["shots"] for i in grouped_event_list]
    
#-------------    
    
shots = pd.read_pickle('data\shot_data.pkl')


##############################################################################################
                       # PRE-PROCESSING #
##############################################################################################
                       
#expand out the nested json values within shots into new dataframe
expanded_shots = json_normalize(shots['shot']) # expanded shot is produced for each shot in order, so can reset index and append shots.id onto expanded_shots

shots = shots.reset_index(drop=True)
expanded_shots = expanded_shots.reset_index(drop=True)

#concat along the columns axis as data is in order
expanded_shots = pd.concat([shots, expanded_shots], axis = 1)

#expad location data into own columns
expanded_shots[['loc_x','loc_y']] = pd.DataFrame(expanded_shots.location.tolist(), index= expanded_shots.index)
expanded_shots[['endloc_x', 'endloc_y', 'endloc_z']] =  pd.DataFrame(expanded_shots.end_location.tolist(), index= expanded_shots.index)

#lookup the date of the match for each shot in the set
match_date = GetMatchDates()
expanded_shots = expanded_shots.merge(match_date) # get date of all shots

#select some players for evaluation - note for this set there is only a handful of players with >100 shots. 
player_list = ['Bojan Krkíc Pérez', 'Philippe Coutinho Correia', 'Alexis Alejandro Sánchez Sánchez']

#create a dictionary for the shot data of the evaluation players
eval_dict = {}
for player in player_list:
    eval_dict[player] = expanded_shots.loc[expanded_shots['player'].isin([player])].sort_values(by = ['match_date', 'index']) # strip out eval players shots and sort into order shots were taken
    
#drop this data from the main set to be used for model training and test
expanded_shots = expanded_shots.loc[~expanded_shots['player'].isin(player_list)]

#generate IO for models - the output is common for both
main_psxg_input, main_xg_input, main_output = PrepareIO(expanded_shots)

#get eval player data into format that can be plugged into model as well
eval_data = {}
for player in eval_dict:
    eval_data[player] = PrepareIO(eval_dict[player])

    

##############################################################################################
                       # MODELS #
##############################################################################################
                       

#----------psxG model----------------------------------------------

#generate train and test split for single run
X_train, X_test, y_train, y_test = train_test_split(main_psxg_input, main_output, test_size=0.3, random_state=42)                     

#define xgboost model - params have been chosen following a small amount of experimentation i.e. NOT optimised
psxg_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=180, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

#fit model
psxg_model.fit(X_train, y_train)

#generate predictions from test data
y_pred = psxg_model.predict_proba(X_test)                       
     
#calculate log loss on test data, using p(Goal = 1) i.e. the second value in the array
ll_model = log_loss(y_test, y_pred[:,1]) 

#plot calibration curve
ccurve = calibration_curve(y_test, y_pred[:,1], n_bins = 15) # returns true proportion [0] and average predicted prob [1]
plt.scatter(ccurve[1],ccurve[0])
plt.title('psxG Calibration Curve')
plt.xlabel('Average of model predicted psxG')
plt.ylabel('Average of actual goal outcome')
x = [0,1]
y = [0,1]
plt.plot(x,y, '--')
plt.show()
#------------------------------------------------------------------

#----------xG model----------------------------------------------

#generate train and test split for single run
X_train, X_test, y_train, y_test = train_test_split(main_xg_input, main_output, test_size=0.3, random_state=42)

#define xgboost model - params have been chosen following a small amount of experimentation i.e. NOT optimised
xg_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=180, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

#fit model
xg_model.fit(X_train, y_train)

#generate predictions from test data
y_pred = xg_model.predict_proba(X_test)  

#calculate log loss on test data, using p(Goal = 1) i.e. the second value in the array
ll_model = log_loss(y_test, y_pred[:,1])    

#plot calibration curve
ccurve = calibration_curve(y_test, y_pred[:,1], n_bins = 15) # returns true proportion [0] and average predicted prob [1]
plt.scatter(ccurve[1],ccurve[0])
plt.title('xG Calibration Curve')
plt.xlabel('Average of model predicted xG')
plt.ylabel('Average of actual goal outcome')
x = [0,1]
y = [0,1]
plt.plot(x,y, '--')
plt.show()  

##############################################################################################
                       # EVALUATION #
##############################################################################################
# Generate confidence intervals on eval player set

window = 30 # number of shots to use as window for rolling average

#initialise an empty dictionary for player results to go in
player_eval = {}
for player in player_list:
    player_eval[player] = {}
    player_eval[player]['psxG_preds'] = []
    player_eval[player]['xG_preds'] = []
    
    
num_sims = 1000 # set number of runs of model fitting to perform - EACH SIM TAKES 2 SECONDS ON MY MID-SPEC MACHINE = 2000 SECS TOTAL

#run the model fits and add predition results to dictionary
for i in range(num_sims): 
    X_train_psxG, X_test_psxG, y_train, y_test = train_test_split(main_psxg_input, main_output, test_size=0.3) 
    
    X_train_xG = X_train_psxG.drop(['angle', 'elev', 'vel'], axis = 1)
    X_test_xG = X_test_psxG.drop(['angle', 'elev', 'vel'], axis = 1)

    psxg_model.fit(X_train_psxG, y_train)
    psxg_pred = psxg_model.predict_proba(X_test_psxG)
    
    xg_model.fit(X_train_xG, y_train)
    xg_pred = xg_model.predict_proba(X_test_xG)
    
    ll_psxg = log_loss(y_test, psxg_pred[:,1]) 
    ll_xg = log_loss(y_test, xg_pred[:,1]) 
    
    print('psxG = '+ str(ll_psxg)) # print progress LL for each run as we go to show it is doing something more than anything
    print('xG = '+ str(ll_xg))
    
    for player in player_list:

        player_eval[player]['psxG_preds'].append(psxg_model.predict_proba(eval_data[player][0])[:,1])
        player_eval[player]['xG_preds'].append(xg_model.predict_proba(eval_data[player][1])[:,1])
        
#manipulate and perform relative % calculation
for player in player_eval:
    player_eval[player]['p'] = np.array(player_eval[player]['psxG_preds'])
    player_eval[player]['q'] = np.array(player_eval[player]['xG_preds'])
    
    p = player_eval[player]['p']
    q = player_eval[player]['q']
    
    player_eval[player]['overperf_mult'] = (p-q) / q
    player_eval[player]['overperf_mult'] = player_eval[player]['overperf_mult'].T
    player_eval[player]['overperf_mult'] = pd.DataFrame(player_eval[player]['overperf_mult'])
    
    player_eval[player]['ma_overperf_mult'] =  player_eval[player]['overperf_mult'].rolling(window = window).mean()
    
    #calculate relavent percentiles for mean and CI of choice - here we have 95% CI
    player_eval[player]['ma_overperf_mult_97'] = np.percentile(player_eval[player]['ma_overperf_mult'], q = 97.5, axis = 1)
    player_eval[player]['ma_overperf_mult_2'] = np.percentile(player_eval[player]['ma_overperf_mult'], q = 2.5, axis = 1)
    player_eval[player]['ma_overperf_mult_50'] = np.percentile(player_eval[player]['ma_overperf_mult'], q = 50, axis = 1)


##############################################################################################
                       # PLOTTING #
##############################################################################################
                       
player = 'Alexis Alejandro Sánchez Sánchez'

plt.figure(figsize=(12,6))
plt.plot(player_eval[player]['ma_overperf_mult_50']*100, label = 'psxG - xG', color = 'purple', linewidth = 5)
plt.plot(player_eval[player]['ma_overperf_mult_97']*100, '--', color = 'purple', label = '95% CI')
plt.plot(player_eval[player]['ma_overperf_mult_2']*100, '--', color = 'purple')

date_labels = eval_dict[player]['match_date']
plt.title('Relative psxG overperformance - '+str(window)+ ' shot rolling average'+'\n'+ player)
plt.xlabel('Date')
plt.ylabel('Relative overperformance %')
plt.legend(loc="upper right")
ticks = np.floor(np.linspace(0,len(date_labels)-1,num = 7))
plt.xticks(ticks = ticks, labels = date_labels.take(ticks))
plt.grid(b=True)
plt.xlim(window, len(date_labels))
plt.ylim(-40,80)

plt.fill_between(x = list(range(0,len(date_labels))), y1 = player_eval[player]['ma_overperf_mult_97']*100,
                 y2 = player_eval[player]['ma_overperf_mult_2']*100,
                 alpha = 0.3,
                 color = 'purple')
plt.show()

