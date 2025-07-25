import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy.spatial.distance import pdist, squareform
import sys
from geopy.distance import geodesic

# 设置系统路径
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

# 导入必要的模块
from SCBIRL_Global_PE.utils import UserDataPart
from SCBIRL_Global_PE.EnDecoder import globalPE
from SCBIRL_Global_PE.gridAttn import GridCellPositionalEncoding
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.transformer as Transformer
import SCBIRL_Global_PE.utils as SIRLU
import jax
import jax.numpy as jnp
import haiku as hk

def load_user_coordinates(max_users = 20):
    """
    从UserDataPart路径下加载用户的访问坐标，使用id_coords_mapping.pkl文件
    
    参数:
    max_users (int): 要处理的最大用户数
    
    返回:
    coords_list (list): 用户坐标列表
    all_coords (np.array): 所有用户坐标合并为一个N*2数组
    """
    print("加载用户坐标数据...")
    
    # 获取用户文件夹
    who_list = [d for d in os.listdir(UserDataPart) if d.isdigit()]
    
    # 限制用户数量
    if max_users:
        who_list = who_list[:max_users]
    
    coords_list = []
    all_coords = []
    
    for who in tqdm(who_list):
        try:
            # 加载id_coords_mapping.pkl文件
            mapping_file = os.path.join(UserDataPart, who, 'id_coords_mapping.pkl')
            if not os.path.exists(mapping_file):
                print(f"找不到文件: {mapping_file}")
                continue
                
            # 加载坐标映射
            with open(mapping_file, 'rb') as f:
                id_coords_mapping = pickle.load(f)
            
            # 提取所有坐标
            user_coords = list(id_coords_mapping.values())
            coords_list.append(user_coords)
            all_coords.extend(user_coords)
        except Exception as e:
            print(f"处理用户 {who} 数据时出错: {e}")
    
    # 转换为numpy数组并去重
    all_coords = np.array(all_coords)
    all_coords_array = np.unique(all_coords, axis=0)
    
    print(f"共加载 {len(coords_list)} 个用户，总计 {len(all_coords_array)} 个唯一坐标")
    
    return coords_list, all_coords_array

def calculate_geographic_distance(coords):
    """
    计算坐标之间的真实地理距离（基于地球表面的大圆距离）
    
    参数:
    coords (np.array): N*2数组，每行是一个经纬度坐标 [lon, lat]
    
    返回:
    distances (np.array): N*N距离矩阵，单位为千米
    """
    print("计算地理距离...")
    
    # 初始化距离矩阵
    n = len(coords)
    distances = np.zeros((n, n))
    
    # 计算距离矩阵（上三角部分）
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            # geodesic函数需要(lat, lon)格式，而我们的数据是(lon, lat)，所以需要反转
            coord_i = (coords[i][1], coords[i][0])  # 转换为(lat, lon)
            coord_j = (coords[j][1], coords[j][0])  # 转换为(lat, lon)
            
            # 计算地理距离（千米）
            distances[i, j] = geodesic(coord_i, coord_j).kilometers
    
    # 填充下三角（距离矩阵是对称的）
    distances = distances + distances.T
    
    return distances

def calculate_absolute_encoding_distance(coords, num_heads=1, num_scale=4):
    """
    计算绝对位置编码之间的距离，使用与模型一致的配置
    
    参数:
    coords (np.array): N*2数组，每行是一个经纬度坐标 [lon, lat]
    num_heads (int): 注意力头数量，默认为6
    num_scale (int): 尺度数量，默认为4
    
    返回:
    encodings (np.array): 位置编码数组
    distances (np.array): N*N距离矩阵
    """
    print("计算绝对位置编码距离...")
    
    # 计算编码维度 - 与模型配置保持一致
    dimension = 2 * num_heads * num_scale
    
    # 计算每个坐标的绝对位置编码
    encodings = []
    for coord in tqdm(coords):
        # 调用globalPE函数计算绝对位置编码
        pe = globalPE(coord, dimension).flatten()
        # 将复数编码转换为实数向量（实部和虚部相加，与模型中方式保持一致）
        pe_flat = pe.real + pe.imag
        encodings.append(pe_flat)
    
    encodings = np.array(encodings)
    
    # 计算编码之间的欧氏距离
    distances = squareform(pdist(encodings, metric='euclidean'))
    
    return encodings, distances


def encoder_inpect(inputs, positions, num_layers, num_heads, num_scale, dff, rate, output_dim, rng):
    # Combine inputs and pe code
    traj_n, pair_n, state_n, position_dim = positions.shape
    embedding_dim = 2 * (position_dim + 1) * num_scale * num_heads
    feature_embedding_layer = hk.Linear(embedding_dim)
    inputs = feature_embedding_layer(inputs)
    
    x = inputs
    
    # Initialize transformer layer
    transformer_layers = [Transformer.TransformerLayer(embedding_dim, num_heads, dff, use_rotation=True, rate=rate) 
                        for _ in range(num_layers)]
    # forward 
    for i, layer in enumerate(transformer_layers):
        x = layer(x, positions, rng)
        if i == 0:
            block_output = x
    
    def open_rot_attention(layer: Transformer.TransformerLayer, inputs, positions, rng):
        position_dim = positions.shape[-1]
        # here the shape turns to (batch_size, 2 * seq_len, 2)
        
        batch_size, _, _, _ = inputs.shape
        query = layer.mha.wq(inputs)
        key = layer.mha.wk(inputs)
        value = layer.mha.wv(inputs)
        # here the shape is (batch_size, num_heads, 2 * seq_len, depth)
        query = layer.mha.split_heads(query, batch_size) 
        key = layer.mha.split_heads(key, batch_size)
        value = layer.mha.split_heads(value, batch_size)
        positions = positions.reshape(batch_size, -1, position_dim)     
        attn_rng, _ = jax.random.split(rng)
        query_rot, key_rot = layer.mha.grid_pe(positions, query, key, attn_rng)
        attention_output, attention_weights = layer.mha.attention(query_rot, key_rot, value)
        return attention_output, attention_weights
    
    attention_output, attention_weights = open_rot_attention(transformer_layers[0], inputs, positions, rng)

    final_layer = hk.Linear(output_dim)
    final_output = final_layer(x)
    return final_output, block_output


def calculate_relative_encoding_distance(coords, who, num_layers=2, num_heads=1, num_scale=4, dff=2, rate=0.1, encoder_o_dim=2):    
    # 初始化模型
    model_dir = f'./model/model_training_weekend_with_prior/'
    who_fold = SIRLU.toWhoString(who)
    final_model_path = os.path.join(model_dir, who_fold, 'evolution_model')
    model_name = sorted(os.listdir(final_model_path))[-1]
    model_param_path = os.path.join(final_model_path, model_name)

    # data_dir = UserDataPart + '{:09d}/'.format(who)
    # iter_start_date = SIRLU.load_traveler(who).iter_start_date
    # inputs, targets_action, positions, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    # model = SIRLT.avril(inputs, targets_action, positions, state_dim, action_dim, state_only=True, coords_proj=mapping_dict)
    
    e_params, q_params = SIRLU.load_pickle_binary(model_param_path)
    expand_dim_linear_params = e_params['linear']
    query_params = e_params['transformer_layer/~/multi_head_self_grid_attention/~/linear']
    key_params = e_params['transformer_layer/~/multi_head_self_grid_attention/~/linear_1']

    state_attrs = SIRLU.load_state_attrs(who=who)
    traj_list = SIRLU.loadTravelChainAll(who)
    fnid_visits = []
    for tc in traj_list:
        fnid_visits.extend(tc.fnid_chain)
    counter = Counter(fnid_visits)
    counter_dict = dict(counter)
    count_series = pd.Series(counter_dict, name='frequency')
    state_attrs.set_index('fnid', inplace=True)
    state_attrs_with_fre = pd.merge(state_attrs, count_series, how='left', left_index=True, right_index=True)
    # 选择除了frequency的列
    select_columns = [c for c in state_attrs_with_fre.columns if c != 'frequency']
    state_attrs_array = state_attrs_with_fre.loc[:, select_columns].to_numpy()
    state_attrs_freq = state_attrs_with_fre['frequency'].to_numpy()
    # calculate the weighted average of the state_attrs_array
    state_attrs_average = np.average(state_attrs_array, weights=state_attrs_freq, axis=0)
    input_array = jnp.expand_dims(state_attrs_average, axis=0)
    
    embeddings = jnp.dot(input_array, expand_dim_linear_params['w']) + expand_dim_linear_params['b']
    embedding_array = jnp.repeat(embeddings, len(coords), axis=0)
    embedding_dim = embedding_array.shape[1]
    
    queries = jnp.dot(embedding_array, query_params['w']) + query_params['b']
    keys = jnp.dot(embedding_array, key_params['w']) + key_params['b']
    # keys = queries.copy()
    keys = keys[None, None, :, :]; queries = queries[None, None, :, :]
    coords = coords[None, :, :]
    
    def grid_pe_forward(coords, queries, keys, num_heads, embedding_dim, rng):
        grid_pe = GridCellPositionalEncoding(
            dimension=2,
            qk_dim=embedding_dim,
            num_heads=num_heads
        )
        return grid_pe(coords, queries, keys, rng)
    
    rng = jax.random.PRNGKey(41310)
    attn_rng, ffn_rng = jax.random.split(rng)
    grid_pe_forward_compile = hk.transform(grid_pe_forward)
    params = grid_pe_forward_compile.init(attn_rng, coords, queries, keys, num_heads, embedding_dim, attn_rng)
    query_rot, key_rot = grid_pe_forward_compile.apply(params, attn_rng, coords, queries, keys, num_heads, embedding_dim, attn_rng)    # 应用网格细胞位置编码
    matmul_qk = jnp.matmul(query_rot, jnp.swapaxes(key_rot, -1, -2))
    scaled_attention = matmul_qk / np.sqrt(embedding_dim)
    
    attention = np.squeeze(jax.device_get(scaled_attention))
    return attention


    
    # print("计算相对位置编码距离...")
    
    # state_dim = 10
    # key = jax.random.PRNGKey(41310)
    # # 准备坐标输入
    # positions = np.array(coords)[None, :, None, :]  # [1, N, 1, 2]
    # inputs = 0.5 * np.ones((1, len(coords), 1, state_dim))  # 创建全1输入向量
    
    # with open(model_param_path, 'rb') as f:    
    #     params = pickle.load(f) 
    #     e_params, _, _ = params

    # encoder_inspect_compile = hk.transform(encoder_inpect)
    # # apply the encoder_inspect_compile to the inputs
    # encoder_inspect_output, representation = encoder_inspect_compile.apply(
    #     e_params, 
    #     key,
    #     inputs, 
    #     positions,
    #     num_layers,
    #     num_heads,
    #     num_scale,
    #     dff,
    #     rate,
    #     encoder_o_dim,
    #     key)
    # # 将修改后的函数转换为Haiku变换
    
    # features = np.squeeze(encoder_inspect_output)
    # # 计算编码之间的欧氏距离
    # distances = squareform(pdist(features, metric='euclidean'))
    # # calculate the inner product of the encoding vector
    # # inner_product = np.matmul(features, features.T)
    
    # return features, distances, representation


if __name__ == "__main__":
    who = 1102234
    
    import geopandas as gpd
    path = './data/city_grid_features/city_grid_features.geojson'
    city_grid_with_LU = gpd.read_file(path)
    city_grid_location = city_grid_with_LU.geometry.centroid
    # convert the coordinates to UTM projection
    city_grid_location = city_grid_location.to_crs(epsg=32650)
    city_grid_location_array = np.array([(p.x, p.y) for p in city_grid_location])
    coords = city_grid_location_array[np.random.choice(len(city_grid_location_array), 8), :]
    
    
    calculate_relative_encoding_distance(coords, who, num_layers=2, num_heads=1, num_scale=4, dff=2, rate=0.1, encoder_o_dim=2)