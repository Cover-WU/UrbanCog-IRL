import jax.numpy as np

import numpy as onp
import pickle
import os
import copy 
import pandas as pd
from tqdm import tqdm

from .utils import getActionDim, loadJsonFile, loadTravelDataFromDicts, preprocessStateAttributes, plugInDataPair, extract_week_ends
from scipy.special import softmax

def getComputeFunction(model, attribute_type):
    """
    Return the appropriate function for model computation.
    attribute_type should be either 'value', 'reward', or 'transition_prob'.
    
    The fed grid code should be complex array.
    """
    if attribute_type == 'value':
        return lambda state,positions: np.max(model.QValue(state, positions))
    elif attribute_type == 'transition_prob':
        return lambda state,positions: softmax(model.QValue(state, positions)[0][0])
    elif attribute_type == 'reward':
        return lambda state,positions: model.reward(state, positions)
    elif attribute_type == 'reward_single':
        return lambda state,positions: model.reward(state, positions)[0][0][0]
    else:
        raise ValueError("attribute_type should be either 'value', 'reward', or 'transition_prob'.")


def readAndPrepareData(user_data_path, start_date):
    """
    Reads and prepares data. 
    Return visited coordinates, divide the data into two parts, and preprocess state attributes.

    Args:
        user_data_path (str): The path to the user data.
        start_date (str): The start date for filtering travel chains.

    Returns:
        - visitedState (set): A set of visited coordinates before start date
        - trajInitChains/trajIterChains (list): A list of travel chains before/after migration.
        - stateAttribute (pd.DataFrame): Dataset of preprocessed state attributes.
    """
    all_traj_path = user_data_path + 'all_traj.json'
    all_traj_feature_path = user_data_path + 'all_traj_feature.csv'
    
    # Load and process the data of travel chains before migration.
    chains = loadJsonFile(all_traj_path)
    beforeChains = [chain for chain in chains if chain['date'] < start_date]
    visitedState = {tuple(state) for chain in beforeChains for state in chain['travel_chain']}
    trajInitChains = loadTravelDataFromDicts(beforeChains)

    # Load and process the data of travel chains after migration.
    afterChains = [chain for chain in chains if chain['date'] >= start_date]
    trajIterChains = loadTravelDataFromDicts(afterChains)

    # Preprocess state attributes based on the after migration data.
    stateAttribute, _ = preprocessStateAttributes(all_traj_feature_path)

    return visitedState, trajInitChains, trajIterChains, stateAttribute

def afterMigrt(model, dataPath, outputPath, start_date, iter_type, prior_iter=True, by_weekend=False):
    '''
    Iteratively train the model with real traj data.
    '''
    
    if prior_iter:
        assert iter_type in ['recent', 'incremental'], "Argument `iter_type` should be either 'recent' or 'incremental'."
        if iter_type == 'recent':
            model_tag = 'iterated'
            folder_name = "evolution_model/"
        else:
            model_tag = 'increased'
            folder_name = 'empirical_model/'
    else:
        model_tag = 'ignorant'
        folder_name = 'no_prior_model/'
    
    full_traj_path = dataPath + "all_traj.json"
    # Load the mapping between IDs and their corresponding fnid.
    with open(dataPath + "id_coords_mapping.pkl", "rb") as f:
        id_coords = pickle.load(f)
    with open(dataPath + "coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)

    all_chains = loadTravelDataFromDicts(loadJsonFile(full_traj_path))
    actionDim = getActionDim(all_chains)

    # Read and preprocess data for analysis.
    visitedState, trajInitChains, trajIterChains, stateAttribute = readAndPrepareData(dataPath, start_date)


    modelDir = outputPath + folder_name
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)
    
    current_model = model
    if by_weekend:
        date_list = [tc.date for tc in trajIterChains]
        weekends, week_codes = extract_week_ends(date_list)
        week_codes_unique = list(set(week_codes))

        for w in sorted(week_codes_unique):
            if not prior_iter:
                current_model = copy.deepcopy(model)
            iter_training_set = [trajIterChains[i] for i, week in enumerate(week_codes) if week == w]

            # Process and calculate reward values after migration.
            plugInDataPair(iter_training_set, stateAttribute, current_model, visitedState)

            # Train the model.
            # change
            # weights = [1 / 2 ** (memory_buffer - i) for i in range(memory_buffer)]
            weights = None
            current_model.train(iters=1000, loss_threshold=0.005, weights=weights, prior=True)

            # Save the current model state.
            modelSavePath = modelDir + model_tag + '_model_' + str(iter_training_set[-1].date) + ".pickle"
            current_model.modelSave(modelSavePath)
        
    else:
        memory_buffer = 10 - 1 
        for i in range(0, len(trajIterChains), 1):
            if not prior_iter:
                current_model = copy.deepcopy(model)
                
            if i < memory_buffer:
                iter_training_set = trajInitChains[-(memory_buffer-i):] + trajIterChains[:i]
            else:
                iter_training_set = trajIterChains[i-memory_buffer:i]
            iter_training_set = iter_training_set + [trajIterChains[i]]

            # Process and calculate reward values after migration.
            plugInDataPair(iter_training_set, stateAttribute, model, visitedState)

            # Train the model.
            # change
            # weights = [1 / 2 ** (memory_buffer - i) for i in range(memory_buffer)]
            weights = None
            model.train(iters=1000, loss_threshold=0.005, weights=weights, prior=True)

            # Save the current model state.
            modelSavePath = modelDir + model_tag + '_model_' + str(iter_training_set[-1].date) + ".pickle"
            model.modelSave(modelSavePath)


# def afterMigrtWeekend(model, dataPath, outputPath, start_date, iter_type):
#     '''
#     Iteratively train the model with real traj data.
#     '''
#     assert iter_type in ['recent', 'prior'], "Argument `iter_type` should be either 'recent' or 'prior'."
#     if iter_type == 'recent':
#         model_tag = 'iterated'
#         folder_name = "evolution_model/"
#     else:
#         model_tag = 'increased'
#         folder_name = 'empirical_model/'
    
#     full_traj_path = dataPath + "all_traj.json"

#     # Load the mapping between IDs and their corresponding fnid.
#     with open(dataPath + "id_coords_mapping.pkl", "rb") as f:
#         id_coords = pickle.load(f)
#     with open(dataPath + "coords_fnid_mapping.pkl", "rb") as f:
#         coords_fnid = pickle.load(f)

#     all_chains = loadTravelDataFromDicts(loadJsonFile(full_traj_path))
#     actionDim = getActionDim(all_chains)

#     # Read and preprocess data for analysis.
#     visitedState, trajInitChains, trajIterChains, stateAttribute = readAndPrepareData(dataPath, start_date)

#     # Initialize an empty DataFrame with predefined columns
#     resultsDf = pd.DataFrame(columns=['coords', 'fnid'])
#     # Iterate over the coords_fnid dictionary and append each key-value pair to resultsDf
#     for key, value in coords_fnid.items():
#         # Append the key-value pair as a new row to resultsDf
#         resultsDf = resultsDf._append({'coords': key, 'fnid': value}, ignore_index=True)


#     modelDir = outputPath + folder_name
#     if not os.path.exists(modelDir):
#         os.makedirs(modelDir)

#     date_list = [tc.date for tc in trajIterChains]
#     weekends, week_codes = extract_week_ends(date_list)
#     week_codes_unique = list(set(week_codes))

#     for w in sorted(week_codes_unique):
#         iter_training_set = [trajIterChains[i] for i, week in enumerate(week_codes) if week == w]

#         # Process and calculate reward values after migration.
#         plugInDataPair(iter_training_set, stateAttribute, model, visitedState)

#         # Train the model.
#         # change
#         # weights = [1 / 2 ** (memory_buffer - i) for i in range(memory_buffer)]
#         weights = None
#         model.train(iters=1000, loss_threshold=0.005, weights=weights, prior=True)

#         # Save the current model state.
#         modelSavePath = modelDir + model_tag + '_model_' + str(iter_training_set[-1].date) + ".pickle"
#         model.modelSave(modelSavePath)
        

def iterative_model_training(
    model,
    dataPath,
    outputPath,
    start_date,
    prior_iter=True,
    by_weekend=False,
    iter_type='recent',
    skip_step=None,
    memory_span=10,
    train_iters=1000,
    loss_threshold=0.005
):
    """
    统一的模型迭代训练函数。
    参数：
        model: 初始模型（有先验时为原始模型，无先验时为tabula rasa模型）
        dataPath: 用户数据路径
        outputPath: 输出路径
        start_date: 迁移起始日期
        prior_iter: 是否有先验迭代（True=有先验，False=无先验）
        by_weekend: 是否按周迭代（True=按周，False=按天）
        iter_type: 'prior' 或 'recent'，决定模型保存路径和tag
        memory_buffer: 记忆窗口长度
        train_iters: 每次训练迭代次数
        loss_threshold: 训练loss阈值
    """
    import copy
    import pandas as pd
    import os
    
    # 数据加载
    full_traj_path = dataPath + "all_traj.json"
    with open(dataPath + "id_coords_mapping.pkl", "rb") as f:
        id_coords = pickle.load(f)
    with open(dataPath + "coords_fnid_mapping.pkl", "rb") as f:
        coords_fnid = pickle.load(f)
    all_chains = loadTravelDataFromDicts(loadJsonFile(full_traj_path))
    actionDim = getActionDim(all_chains)
    visitedState, trajInitChains, trajIterChains, stateAttribute = readAndPrepareData(dataPath, start_date)

    # 结果目录和tag
    if prior_iter:
        if iter_type == 'recent':
            model_tag = 'iterated'
            folder_name = "evolution_model/"
        elif iter_type == 'incremental':
            model_tag = 'increased'
            folder_name = 'empirical_model/'
        else:
            raise ValueError("iter_type should be either 'recent' or 'prior'.")
    else:
        model_tag = 'ignorant'
        folder_name = 'no_prior_model/'
    modelDir = outputPath + folder_name
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    # 生成迭代单元
    if by_weekend:
        date_list = [tc.date for tc in trajIterChains]
        weekends, week_codes = extract_week_ends(date_list)
        week_codes_unique = list(set(week_codes))
        iter_units = []
        for w in sorted(week_codes_unique):
            indices = [i for i, week in enumerate(week_codes) if week == w]
            iter_units.append([trajIterChains[i] for i in indices])
        iter_unit_dates = [unit[-1].date for unit in iter_units]
    else:
        iter_units = []
        memory_buffer = memory_span - 1
        skip = skip_step if skip_step is not None else 1
        for i in range(0, len(trajIterChains), skip):
            if i < memory_buffer:
                iter_training_set = trajInitChains[-(memory_buffer-i):] + trajIterChains[:i]
            else:
                iter_training_set = trajIterChains[i-memory_buffer:i]
            iter_training_set = iter_training_set + [trajIterChains[i]]
            iter_units.append(iter_training_set)
        iter_unit_dates = [unit[-1].date for unit in iter_units]

    # 迭代训练
    current_model = model
    for idx, iter_training_set in enumerate(iter_units):
        # 选择模型
        if not prior_iter:
            current_model = copy.deepcopy(model)
        # 数据注入
        plugInDataPair(iter_training_set, stateAttribute, current_model, visitedState)
        # 训练
        current_model.train(iters=train_iters, loss_threshold=loss_threshold, weights=None, prior=prior_iter)
        # 保存
        modelSavePath = modelDir + f'{model_tag}_model_{iter_unit_dates[idx]}.pickle'
        current_model.modelSave(modelSavePath) 
        