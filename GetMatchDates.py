# -*- coding: utf-8 -*-
"""

@author: charles william

"""
#utility function to get the dates that an event took place by looking up the match ID


from statsbombpy import public
import pandas as pd
from pandas.io.json import json_normalize
from itertools import chain



def GetMatchDates():
    
    #### CREATE COMPETITIONS TABLE ####
    
    comps = public.competitions()
    comp_frame = pd.DataFrame.from_dict(comps, orient = 'index') 
    comp_frame['comp_seas'] = comp_frame['competition_id'].astype(str) +'_'+comp_frame['season_id'].astype(str) # concat comp and seas ids and set as index
    comp_frame = comp_frame.set_index('comp_seas') # **** UPLOAD ******
    
    
    ###################################
    
    #### CREATE MATCHES TABLE ####
    
    matches_dictlist=[] # returns list of dict of dicts
    for i in range(0, len(comp_frame.index)):
        matches = public.matches(comp_frame['competition_id'][i], comp_frame['season_id'][i])
        matches_dictlist.append(matches)
    
    
    match_list = [] # convert to list of list of dicts
    for dicts in matches_dictlist:
        list_of_dicts = [value for value in dicts.values()]
        match_list.append(list_of_dicts)
        
    
    match_frame = pd.DataFrame(list(chain.from_iterable(match_list))) 
    
    cols = ['competition',
            'season',
            'home_team',
            'away_team',]
    
    list_of_exp_frames = []
    for col in cols:
        df = json_normalize(match_frame[col])
        list_of_exp_frames.append(df)
        
    expanded_match = pd.concat(list_of_exp_frames, axis = 1)    
    match_frame = pd.concat([match_frame, expanded_match], axis = 1)
    match_frame['comp_seas'] =  match_frame['competition_id'].astype(str) +'_'+match_frame['season_id'].astype(str) # concat comp and seas ids for use as foreign key
    #drop  compeitition, season, home_team, away_team, match_status, metadata, competition_stage, stadium, referee, home_team_group, country.id, country.name, away_team_group, managers
    
    match_frame = match_frame.loc[:,~match_frame.columns.duplicated()] # drop duplicated cols (both)
    match_frame = match_frame.drop(['competition',
                                    'season',
                                    'home_team',
                                    'away_team',
                                    'match_status',
                                    'metadata',
                                    'competition_stage',
                                    'stadium',
                                    'referee',
                                    'home_team_group',
                                    'away_team_group',
                                    'managers'], axis = 1) 
    
    ###############################
    
    
    match_date = match_frame.loc[:,["match_id","match_date"]]
    return match_date      