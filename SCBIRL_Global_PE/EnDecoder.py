import haiku as hk

import jax.numpy as np
import numpy as onp
import pyproj

from time import time
from .transformer import *
Q = np.load('./data/Q_matrix.npy')

def compress_pe_code_complex(pe_code, target_dim):
    pe_code_real = np.real(pe_code)
    pe_code_imag = np.imag(pe_code)

    # comprass real and imag
    linear_pe = hk.Linear(output_size=target_dim)
    
    # reshape concatenated_pe_code
    original_shape = pe_code_real.shape[:-1]
    depth = pe_code_real.shape[-1]

    pe_real_compressed = linear_pe(pe_code_real.reshape(-1, depth))
    pe_imag_compressed = linear_pe(pe_code_imag.reshape(-1,depth))
    
    new_shape = original_shape + (target_dim,)
    pe_real_compressed = pe_real_compressed.reshape(*new_shape)
    pe_imag_compressed = pe_imag_compressed.reshape(*new_shape)

    return pe_real_compressed,pe_imag_compressed


def globalPE(coords, dimension, random_rotation=True, seed=43):
    '''
    Calculate positional encoding from coordinates:
    A complex matrix of shape (dimension, 3) is returned.
    First convert lat/lon to UTM coordinates, then apply encoding.
    '''
    x, y = coords
    onp.random.seed(seed)
    
    # 使用相同的角度生成方法
    if random_rotation:
        angle_list = onp.random.uniform(0, 2 * onp.pi, dimension)
    else:
        angle_list = onp.zeros(dimension)
    
    # 预先计算旋转矩阵
    theta = 2 * onp.pi / 3
    R = onp.array([[onp.cos(theta), -onp.sin(theta)], [onp.sin(theta), onp.cos(theta)]])
    
    # 预先计算所有尺度因子
    scale_factors = 1 / (200 ** (onp.arange(dimension) / dimension))
    
    # 常量因子
    omega_factor = (2 * onp.pi / 1000)
    
    # 预计算所有角度的正弦和余弦
    cos_angles = onp.cos(angle_list)
    sin_angles = onp.sin(angle_list)
    
    # 创建所有omega向量
    omega_n0_all = onp.vstack([cos_angles, sin_angles]).T * scale_factors[:, onp.newaxis] * omega_factor
    
    # 计算omega_n1和omega_n2（应用旋转矩阵）
    omega_n1_all = onp.dot(omega_n0_all, R.T)  # 使用R.T进行批量矩阵乘法
    omega_n2_all = onp.dot(omega_n1_all, R.T)
    
    # 坐标向量
    coords_vec = onp.array([x, y])
    
    # 计算点积
    dot_products_0 = onp.dot(omega_n0_all, coords_vec)
    dot_products_1 = onp.dot(omega_n1_all, coords_vec)
    dot_products_2 = onp.dot(omega_n2_all, coords_vec)
    
    # 计算复指数
    eiw0x_all = onp.exp(1j * dot_products_0)
    eiw1x_all = onp.exp(1j * dot_products_1)
    eiw2x_all = onp.exp(1j * dot_products_2)
    
    # 组合所有eiw值
    eiw_all = onp.stack([eiw0x_all, eiw1x_all, eiw2x_all], axis=1)
    
    # 应用Q矩阵
    g_all = onp.dot(eiw_all, Q.T)
    
    return g_all


def generate_random_unitary_matrix(dim, seed=43):
    onp.random.seed(seed)
    A = onp.random.randn(dim, dim) + 1j * onp.random.randn(dim, dim)  # 随机复矩阵
    Q, R = onp.linalg.qr(A)  # QR分解得到酉矩阵
    return Q

def encoder_model(inputs, coords, num_layers, num_heads, num_scale, dff, rate, output_dim, rng):
    # Combine inputs and pe code
    traj_n, pair_n, state_n, coord_dim = coords.shape
    embedding_dim = 2 * (coord_dim + 1) * num_scale * num_heads
    feature_embedding_layer = hk.Linear(embedding_dim)
    inputs = feature_embedding_layer(inputs)
    
    # Compute global PE
    flat_coords = coords.reshape(-1, coord_dim).tolist()
    flat_coords = [tuple(coord) for coord in flat_coords]
    # flat_pe_codes = np.stack([posicode[coord] for coord in flat_coords], axis=0)
    flat_pe_codes = [globalPE(coord, embedding_dim // (coord_dim + 1)).flatten() 
                     for coord in flat_coords]
    pe_code = [code.real + code.imag for code in flat_pe_codes]
    pe_code = pe_code.reshape(traj_n, pair_n, state_n, embedding_dim)  # reshape back
    x = inputs + pe_code
    
    # Initialize transformer layer
    transformer_layers = [TransformerLayer(embedding_dim, num_heads, dff, use_rotation=True, rate=rate) 
                        for _ in range(num_layers)]
    # forward 
    for layer in transformer_layers:
        x = layer(x, coords, rng)

    final_layer = hk.Linear(output_dim)
    final_output = final_layer(x)
    return final_output

def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1)
    mask = mask[np.newaxis, np.newaxis, ...]
    return mask

def q_network_model(inputs, enc_output, coords, num_layers, num_heads, num_scale, dff, rate, output_dim, rng):
    # combine inputs and pe code
    position_dim = positions.shape[-1]
    embedding_dim = 2 * (position_dim + 1) * num_scale * num_heads
    feature_embedding_layer = hk.Linear(embedding_dim)
    inputs = feature_embedding_layer(inputs)
    
    # Compute global PE
    traj_n, pair_n, state_n, coord_dim = positions.shape
    flat_positions = positions.reshape(-1, coord_dim)
    pe_list = []
    for coords in flat_positions:
        pe = globalPE(coords, embedding_dim//3)  # shape: (dimension, 3)
        pe_list.append(pe.flatten())               # shape: (dimension * 3,) == embedding_dim
    pe_array = np.stack(pe_list, axis=0)  # shape: (traj_n * pair_n * 2, embedding_dim)
    pe_code = pe_array.reshape(traj_n, pair_n, state_n, embedding_dim)  # reshape back

    pe_real_code,pe_imag_code = np.real(pe_code), np.imag(pe_code)
    x = inputs + pe_real_code + pe_imag_code
    
    # Initialize transformer decoder layer
    transformer_decoder_layers = [TransformerDecoderLayer(embedding_dim, num_heads, dff, use_rotation=True, rate=rate) 
                                for _ in range(num_layers)]
    
    look_ahead_mask = create_look_ahead_mask(inputs.shape[1]*inputs.shape[2])
    # forward function
    for layer in transformer_decoder_layers:
        x = layer(x, enc_output, look_ahead_mask, None, coords, rng)

    final_layer = hk.Linear(output_dim)
    return final_layer(x)

def klGaussianStandard(mean, var):
    return 0.5 * (-np.log(var) - 1.0 + var + mean ** 2)

def kl_divergence(mean1, stddev1, mean2, stddev2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two one-dimensional Gaussian distributions.

    Args:
        mean1 (float): Mean of the first distribution.
        stddev1 (float): Standard deviation of the first distribution.
        mean2 (float): Mean of the second distribution.
        stddev2 (float): Standard deviation of the second distribution.

    Returns:
        float: The KL divergence value.
    """
    # Calculate the KL divergence
    kl = np.log(stddev2 / stddev1) + ((stddev1 ** 2 + (mean1 - mean2) ** 2) / (2 * stddev2 ** 2)) - 0.5
    return kl
