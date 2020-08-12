# -*- coding: utf-8 -*-
"""

@author: charles william
"""
# Utility function to prepare the inputs and outputs for training and testing of the xG and psxG models

import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scipy.stats import gaussian_kde


def PrepareIO(expanded_shots):
    freeze_frames = expanded_shots.loc[:,['freeze_frame','id']]
    
    #create a list of dataframes containing freeze frame data
    list_of_ffFrames = []
    for frame in freeze_frames.itertuples(index=False):
        if frame[0] != frame[0]: # using the nan != nan trick to find find nans in column of dicts
            continue
        
        df = pd.DataFrame(frame[0])
        df['id'] = frame[1]
        df_player = json_normalize(df['player']) # expand the dict cols, append, then drop the originals
        df_position = json_normalize(df['position'])
        df = df.reset_index(drop=True)
        df_player = df_player.reset_index(drop=True)
        df_position = df_position.reset_index(drop=True)
        df = pd.concat([df, df_player,df_position], axis = 1)
        df = df.drop(['player','position'], axis = 1)
        list_of_ffFrames.append(df)
        
    #combine into one dataframe
    ff_frame = pd.concat(list_of_ffFrames)
    
    #expand location coords into own columsn and drop original
    ff_frame[['loc_x','loc_y']] = pd.DataFrame(ff_frame.location.tolist(), index= ff_frame.index)
    ff_frame = ff_frame.drop('location', axis = 1)  
    
    #assign column names
    ff_frame.columns.values[[1, 2, 3, 4, 5]] = ['shot_id', 'player_id', 'player_name', 'pos_id', 'pos_name']  
    
    #drop all data where teammate = True, so we just have opponent loc data
    ff_frame = ff_frame[ff_frame['teammate']==False]
    
    #get rid of redundant / uninteresting columns
    expanded_shots = expanded_shots.drop(['location', 'end_location', 'index', 'minute', 'second', 'type', 'related_events', 'shot','freeze_frame', 'key_pass_id'], axis = 1)  
    
    #gather location data for the goalkeeper for each shot
    GK_locs = ff_frame[ff_frame['pos_name']=='Goalkeeper']
    
    #give id columns a common name for merging with expanded shots
    GK_locs = GK_locs.rename(columns = {'shot_id': 'id'})
    ff_frame = ff_frame.rename(columns = {'shot_id': 'id'})
    
    
    # we'll do a right join - so shots with no freeze frame or no keeper in frame will be dropped - this is mostly pens and a handful of other shots I'm not sure what's going on
    # 83 out of 7934 dropped
    expanded_shots = pd.merge(expanded_shots, GK_locs, on = 'id', how = 'right')
    
    
    shot_locs = expanded_shots.loc[:,['id', 'loc_x_x', 'loc_y_x']]
    
    ff_frame = pd.merge(ff_frame, shot_locs, on = 'id', how = 'right')
    
    #calculate number of defenders in cone
    l_post = (120,36) # SB post coords
    r_post = (120,44)
    in_cone_list = []
    
    for row in ff_frame.itertuples(index=False): # for each shot in freeze frames dataframe...
        
        if row[5] == 'Goalkeeper': # ignore goalkeeper in count
            in_cone_list.append(False)
        else:    
            shot_x = row[8] #extract locations for cone vertices
            shot_y = row[9]
            def_x = row[6]
            def_y = row[7]
        
            point = Point(def_x, def_y) # point of interest is the the defender location
            polygon = Polygon([(shot_x, shot_y), l_post, r_post]) # define cone polygon
            in_cone = polygon.contains(point) # return true if point is within polygon
            in_cone_list.append(in_cone)
    
    #add a new column for the cone data 
    ff_frame['in_cone'] = in_cone_list
    
    #count number of players in cone for each shot
    cone_count = ff_frame.groupby(['id']).sum()['in_cone']
    cone_count = cone_count.to_frame()
    cone_count['id'] = cone_count.index
    cone_count.index.name = None
    
    #merge this data back into expanded shots
    expanded_shots = pd.merge(expanded_shots, cone_count, on ='id')
    
    #select the columns we are ineterested in for our analysis
    cols = ['id', 'team', 'player', 'duration', 'under_pressure',
            'statsbomb_xg', 'technique.name', 'outcome.name',
            'type.name', 'body_part.name', 'first_time', 'deflected',
            'loc_x_x', 'loc_y_x', 'endloc_x', 'endloc_y', 'endloc_z',
            'loc_x_y', 'loc_y_y', 'in_cone']
    
    #define data to be used for psxG model - which will be a superset of the xG model
    psxg_data = expanded_shots.loc[:,cols]
    
    #rename the annoying default column headers
    psxg_data = psxg_data.rename(columns = {'loc_x_x': 'loc_x','loc_y_x': 'loc_y','loc_x_y': 'GKloc_x','loc_y_y': 'GKloc_y' })
    
    #calculate angle, elevation, and velocity
    def getAngle(df):
        x_diff = df['endloc_x'] - df['loc_x']
        y_diff = df['endloc_y'] - df['loc_y']
        
        if x_diff == 0:
            angle = 1.5708 # 90 deg in radians
        else:    
            angle = np.arctan(y_diff/x_diff)
        return angle
    
    def getVelocity(df):
        x_diff = df['endloc_x'] - df['loc_x']
        y_diff = df['endloc_y'] - df['loc_y']
        
        dist = np.sqrt((x_diff**2) + (y_diff**2))
        vel = dist / df['duration'] # any duration = 0 events will show warning but will clipped to 50m/s below - alternatively you could just strip out
        if vel >= 50: # limit velocity to 50 m/s (112mph)
            vel = 50
        return vel
    
    def getElevation(df):
        
        if np.isnan(df['endloc_z']):
            elev = np.nan
            
        else:
            x_diff = df['endloc_x'] - df['loc_x']
            y_diff = df['endloc_y'] - df['loc_y']
            dist = np.sqrt((x_diff**2) + (y_diff**2))
            
            z_diff = df['endloc_z']
            
            elev = np.arctan(z_diff/dist)
        
        return elev
    
    #define kernel density estimation function for fitting pdf of shot elevation distribution
    def kde(x, x_grid, bandwidth=0.2, **kwargs): # can play around with bandwidth
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)
    
    # sample from our pdf
    def generate_rand_from_pdf(pdf, x_grid, nsamples):
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        values = np.random.rand(nsamples)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = x_grid[value_bins]
        return random_from_cdf
     
    # perfrom post-shot calcs on each shot
    psxg_data['angle'] = psxg_data.apply(getAngle, axis = 1)
    psxg_data['vel'] = psxg_data.apply(getVelocity, axis = 1)
    psxg_data['elev'] = psxg_data.apply(getElevation, axis = 1)
    
    #drop shots where there is no elevation calc due to no z loc so we can fit a distribution function
    elev_data = psxg_data['elev'].dropna() 
    
    #generste a 1000 point pdf using KDE for the distribition of the elevation data
    x_grid = np.linspace(min(elev_data), max(elev_data), 1000)
    kdepdf = kde(elev_data, x_grid, bandwidth=0.1)

    
    #impute missing elevations sampling from the distribution of all shots with elevation, so model can't infer blocked/wayward from elevation
    i = 0
    for row in psxg_data.itertuples(index = False):
        if row[22] != row[22]: # look for nan
            psxg_data['elev'].iloc[i] = generate_rand_from_pdf(kdepdf, x_grid, 1)
            i = i+1
        else:
            i = i+1
            
    # convert outcome into 0,1 - 1 for goal, 0 for everything else
               
    output = psxg_data.loc[:,'outcome.name']
    output = output.replace("Goal",1)
    output = output.where(output == 1, 0) # replace any values that aren't 1 with a 0
    output = output.astype(int)
    
    #create input data for psxG model
    psxg_input = psxg_data.loc[:,['body_part.name','loc_x', 'loc_y', 'GKloc_x', 'GKloc_y', 'in_cone', 'angle', 'vel', 'elev']] 
    
    #prduce Header column - 1 if shot was a header, otherwise 0
    psxg_input['body_part.name'] = psxg_input['body_part.name'].replace("Head",1)
    psxg_input['body_part.name'] = psxg_input['body_part.name'].where(psxg_input['body_part.name']==1,0)
    psxg_input['body_part.name'] = psxg_input['body_part.name'].astype(int)
    psxg_input = psxg_input.rename(columns = {'body_part.name': 'header'})
    
    #create xG model input data which is a subset of the psxG model input
    xg_input = psxg_input.loc[:,['header','loc_x', 'loc_y', 'GKloc_x', 'GKloc_y', 'in_cone']]
    
    
    return psxg_input, xg_input, output
