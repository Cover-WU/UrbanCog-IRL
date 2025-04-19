import os
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
from SCBIRL_Global_PE.utils import globalPE, UserDataPart
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


def calculate_relative_encoding_distance(coords, num_layers=2, num_heads=1, num_scale=4, dff=2, rate=0.1, encoder_o_dim=2):    
    # 初始化模型
    model_param_path = './model/traj_Q4/traj_Q4_model.pickle'
    print("计算相对位置编码距离...")
    
    state_dim = 10
    key = jax.random.PRNGKey(41310)
    # 准备坐标输入
    positions = np.array(coords)[None, :, None, :]  # [1, N, 1, 2]
    inputs = 0.5 * np.ones((1, len(coords), 1, state_dim))  # 创建全1输入向量
    
    with open(model_param_path, 'rb') as f:    
        params = pickle.load(f) 
        e_params, _, _ = params

    encoder_inspect_compile = hk.transform(encoder_inpect)
    # apply the encoder_inspect_compile to the inputs
    encoder_inspect_output, representation = encoder_inspect_compile.apply(
        e_params, 
        key,
        inputs, 
        positions,
        num_layers,
        num_heads,
        num_scale,
        dff,
        rate,
        encoder_o_dim,
        key)
    # 将修改后的函数转换为Haiku变换
    
    features = np.squeeze(encoder_inspect_output)
    # 计算编码之间的欧氏距离
    distances = squareform(pdist(features, metric='euclidean'))
    # calculate the inner product of the encoding vector
    # inner_product = np.matmul(features, features.T)
    
    return features, distances, representation

# def generate_coordinate_grid(center_lon, center_lat, grid_size=10, step_size=0.01):
#     """
#     生成以中心点为基准的经纬度网格
    
#     参数:
#     center_lon (float): 中心点经度
#     center_lat (float): 中心点纬度
#     grid_size (int): 网格大小，生成的网格点数为 grid_size x grid_size
#     step_size (float): 网格步长（度）
    
#     返回:
#     coords (np.array): 网格点坐标，形状为 (grid_size*grid_size, 2)
#     """
#     # 计算网格的起始点
#     start_lon = center_lon - (grid_size // 2) * step_size
#     start_lat = center_lat - (grid_size // 2) * step_size
    
#     # 生成经纬度网格
#     lons = np.linspace(start_lon, start_lon + grid_size * step_size, grid_size)
#     lats = np.linspace(start_lat, start_lat + grid_size * step_size, grid_size)
    
#     # 创建网格点坐标
#     coords = []
#     for lon in lons:
#         for lat in lats:
#             coords.append([lon, lat])
    
#     return np.array(coords)

# def visualize_position_encoding(model_path=None, center_coords=None, grid_size=10, step_size=0.01, save_path=None):
#     """
#     可视化不同位置的编码特征
    
#     参数:
#     model_path (str): 模型参数文件路径
#     center_coords (tuple): 中心点坐标 (lon, lat)，如果为None则使用默认值
#     grid_size (int): 网格大小
#     step_size (float): 网格步长（度）
#     save_path (str): 结果保存路径
#     """
#     import matplotlib.pyplot as plt
#     from sklearn.manifold import TSNE
#     from sklearn.decomposition import PCA
    
#     # 设置默认中心点（北京）
#     if center_coords is None:
#         center_coords = (116.3972, 39.9075)  # 北京天安门坐标 (经度, 纬度)
    
#     print(f"使用中心点坐标: {center_coords}")
    
#     # 生成坐标网格
#     coords = generate_coordinate_grid(center_coords[0], center_coords[1], grid_size, step_size)
#     print(f"生成了 {len(coords)} 个网格点")
    
#     # 计算位置编码特征
#     print("计算位置编码...")
#     features, distances = calculate_relative_encoding_distance(coords, model_path=model_path)
    
#     # 降维以便可视化
#     print("使用降维算法进行特征可视化...")
    
#     # 使用PCA
#     pca = PCA(n_components=2)
#     features_pca = pca.fit_transform(features)
    
#     # 使用t-SNE
#     tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(coords)-1))
#     features_tsne = tsne.fit_transform(features)
    
#     # 创建可视化图
#     plt.figure(figsize=(20, 10))
    
#     # 原始坐标网格
#     plt.subplot(2, 3, 1)
#     plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap='viridis', s=50)
#     plt.colorbar(label='坐标索引')
#     plt.title('原始坐标网格')
#     plt.xlabel('经度')
#     plt.ylabel('纬度')
#     plt.grid(True)
    
#     # PCA降维结果
#     plt.subplot(2, 3, 2)
#     scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=np.arange(len(coords)), cmap='viridis', s=50)
#     plt.colorbar(label='坐标索引')
#     plt.title('PCA降维后的特征空间')
#     plt.xlabel('PCA维度1')
#     plt.ylabel('PCA维度2')
#     plt.grid(True)
    
#     # t-SNE降维结果
#     plt.subplot(2, 3, 3)
#     scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=np.arange(len(coords)), cmap='viridis', s=50)
#     plt.colorbar(label='坐标索引')
#     plt.title('t-SNE降维后的特征空间')
#     plt.xlabel('t-SNE维度1')
#     plt.ylabel('t-SNE维度2')
#     plt.grid(True)
    
#     # 中心点与其他点的距离热图
#     plt.subplot(2, 3, 4)
#     center_idx = (grid_size**2) // 2  # 中心点索引
#     center_distances = distances[center_idx]
    
#     # 重塑为网格以便于可视化
#     distance_grid = center_distances.reshape(grid_size, grid_size)
#     plt.imshow(distance_grid, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='与中心点的编码距离')
#     plt.title('中心点与其他点的编码距离热图')
    
#     # 中心点特征与其他点的欧氏距离与地理距离的关系
#     plt.subplot(2, 3, 5)
    
#     # 计算地理距离
#     geo_distances = []
#     center_coord = coords[center_idx]
#     center_latlon = (center_coord[1], center_coord[0])  # 转为(lat, lon)格式
    
#     for coord in coords:
#         latlon = (coord[1], coord[0])  # 转为(lat, lon)格式
#         geo_dist = geodesic(center_latlon, latlon).kilometers
#         geo_distances.append(geo_dist)
    
#     plt.scatter(geo_distances, center_distances, c=np.arange(len(coords)), cmap='viridis', s=50, alpha=0.7)
#     plt.colorbar(label='坐标索引')
#     plt.title('编码距离与地理距离的关系')
#     plt.xlabel('地理距离 (km)')
#     plt.ylabel('编码特征距离')
#     plt.grid(True)
    
#     # 距离分布直方图
#     plt.subplot(2, 3, 6)
#     plt.hist(center_distances, bins=30, color='blue', alpha=0.7)
#     plt.title('与中心点的编码距离分布')
#     plt.xlabel('编码距离')
#     plt.ylabel('频率')
#     plt.grid(True)
    
#     plt.tight_layout()
    
#     # 保存结果
#     if save_path:
#         plt.savefig(save_path)
#         print(f"结果已保存到 {save_path}")
    
#     plt.show()
    
#     # 返回计算结果以便进一步分析
#     results = {
#         'coords': coords,
#         'features': features,
#         'distances': distances,
#         'features_pca': features_pca,
#         'features_tsne': features_tsne,
#         'center_distances': center_distances,
#         'geo_distances': geo_distances
#     }
    
#     return results

if __name__ == "__main__":
    # 测试calculate_relative_encoding_distance函数
    
    # 创建一些示例坐标点
    coords = np.array([
        [116.3972, 39.9075],  # 北京天安门
        [116.4017, 39.9220],  # 故宫
        [116.3906, 39.9133],  # 天安门西
        [116.4041, 39.9010],  # 天安门东
        [116.3980, 39.8985],  # 前门
    ])
    
    print("测试坐标点:")
    for i, coord in enumerate(coords):
        print(f"{i+1}. 经度: {coord[0]}, 纬度: {coord[1]}")
    
    print("\n计算位置编码和距离...")
    features, distances = calculate_relative_encoding_distance(coords)
    
    print(f"\n提取的特征形状: {features.shape}")
    print("特征样本 (第一个坐标点的前10个值):")
    print(features[0, :10])
    
    print("\n距离矩阵:")
    for i in range(len(coords)):
        for j in range(len(coords)):
            print(f"{distances[i, j]:.4f}\t", end="")
        print()
    
    # 计算地理距离以便比较
    print("\n地理距离 (公里):")
    geo_distances = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            coord_i = (coords[i][1], coords[i][0])  # 转为(lat, lon)
            coord_j = (coords[j][1], coords[j][0])  # 转为(lat, lon)
            geo_distances[i, j] = geodesic(coord_i, coord_j).kilometers
            print(f"{geo_distances[i, j]:.4f}\t", end="")
        print()
    
    # 输出编码距离与地理距离的相关性
    from scipy.stats import pearsonr
    
    # 将距离矩阵转为一维数组，忽略对角线(自己到自己的距离)
    dist_pairs = []
    geo_pairs = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):  # 只取上三角矩阵
            dist_pairs.append(distances[i, j])
            geo_pairs.append(geo_distances[i, j])
    
    # 计算相关性
    if len(dist_pairs) > 1:  # 确保有足够的数据点
        correlation, p_value = pearsonr(dist_pairs, geo_pairs)
        print(f"\n编码距离与地理距离的相关性: {correlation:.4f} (p-value: {p_value:.4f})")
        
    print("\n测试完成!")
