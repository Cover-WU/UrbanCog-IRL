# import module from the parent directory
import sys
import os
from tqdm import tqdm

working_directory = os.path.abspath('.')
sys.path.append(working_directory)

import numpy as np
import pandas as pd
import geopandas as gpd

from SCBIRL_Global_PE.utils import UserDataPart, load_fnid_coords_mapping, loadModel, toWhoString
from SCBIRL_Global_PE.migrationProcess import readAndPrepareData

def mentalMap(who, date, model_dir='./model/'):
    model = loadModel(who, date, model_dir=model_dir)
    
    # read the geodataframe
    path = './data/city_grid_features/city_grid_features.geojson'
    city_grid_with_LU = gpd.read_file(path)
    feature_name = ['LU_Business','LU_Green','LU_Industry','LU_Public','LU_Residence','subway',
                    'density','intersections','road_density','rent']
    # read the original built env features
    BE_features = city_grid_with_LU.loc[:, feature_name].to_numpy()
    
    # please note: the feature should be preprocessed 
    # dataPath = UserDataPart + '{:09d}/'.format(who)
    # visit_coords, *_ = readAndPrepareData(dataPath, date)
    # coords_fnid_mapping = load_fnid_coords_mapping(who)
    # visit_fnid = [coords_fnid_mapping[coord] for coord in visit_coords]
    # convert the feature GeoDataFrame to DataFrame by drop the geometry column
    # city_grid_only_features = pd.DataFrame(city_grid_with_LU.drop(columns='geometry'))
    # get the feature values of the visited locations
    # visited_features = city_grid_only_features.loc[city_grid_with_LU.fnid.isin(visit_fnid), feature_name].to_numpy()
    # get the feature minimum and maximum values
    # features_min = visited_features.min(axis=0)
    # features_max = visited_features.max(axis=0)

    path = UserDataPart + toWhoString(who) + '/' + 'all_traj_feature.csv'
    built_attr = pd.read_csv(path)
    features_min, features_max = built_attr[feature_name].min().to_numpy(), built_attr[feature_name].max().to_numpy()
    # normalize the features
    BE_features = (BE_features - features_min) / (features_max - features_min)
    
    # compute the grid centroid and convert it to grid code
    grid_locations = city_grid_with_LU.geometry.centroid
    grid_coords = [(cent.x, cent.y) for cent in grid_locations]
    state_dim = len(feature_name)
    PE_features = np.array(grid_coords, dtype=np.float32)
    PE_features = np.array(PE_features)
        
    # for loop to predict score and save the geodataframe
    BE_arrays = BE_features[:, None, None, :]
    PE_arrays = PE_features[:, None, None, :]
    reward_arrays = model.reward(BE_arrays, PE_arrays)
    reward_arrays = reward_arrays.squeeze()
    city_grid_with_reward = city_grid_with_LU.copy()
    city_grid_with_reward['reward'] = reward_arrays[:, 0]
    return city_grid_with_reward