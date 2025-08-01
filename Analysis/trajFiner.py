# import module from the parent directory
import sys
import os
from tqdm import tqdm
from datetime import date
from math import floor
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import folium
import scipy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
# from hdbscan import HDBSCAN
from sklearn.cluster import HDBSCAN
import jax.numpy as jnp

working_directory = os.getcwd()
working_directory = os.path.abspath('.')
sys.path.append(working_directory)

from Analysis.topoMap import clusterLocations, cogTopoGraph, topoResPath
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import Analysis.performanceEval as Eval

def trajWeekSubset(who, week_order: int):
    visit_dates = SIRLU.visited_date(who)
    traj_total = SIRLU.load_all_traj(who)
    
    _, week_code = SIRLU.extract_week_ends(visit_dates)
    week_order_indices = [i for i, w in enumerate(week_code) if w == week_order]
    if week_order_indices == []:
        return []
    start_date = visit_dates[week_order_indices[0]]
    end_date = visit_dates[week_order_indices[-1]]
    tcs = list(filter(lambda tc: start_date <= tc.date <= end_date, traj_total))
    return tcs

def traj4Visualize(who, week_order: int,):
    tcs = trajWeekSubset(who, week_order)
    od_pairs = []
    for tc in tcs:
        coord_seq = tc.travel_chain
        if len(coord_seq) < 2:
            continue
        for i in range(len(coord_seq) - 1):
            flow = (tuple(coord_seq[i]), tuple(coord_seq[i+1]))
            od_pairs.append(flow)
    return od_pairs
    
def cogGraph4Visualize(who, week_order: int):
    tcs = trajWeekSubset(who, week_order)
    tc_week_end_date = max(tcs, key=lambda tc: tc.date)
    week_end_date = tc_week_end_date.date
    return cogTopoGraph(who, week_end_date)


def eigenDecomposition(A, plot=True):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigenvalues for visual inspection
    :return A tuple containing:
        - the optimal number of clusters by eigengap heuristic
        - all eigenvalues
        - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D^-1/2 A D^-1/2.
    2. Find the eigenvalues and their associated eigen vectors.
    3. Identify the maximum gap which corresponds to the number of clusters
       by eigengap heuristic.

    References:
    - https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    - http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4485%5B0%5D.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # k parameter: Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)

    if plot:
        plt.title("Largest eigenvalues of input matrix")
        plt.scatter(range(len(eigenvalues)), eigenvalues)
        plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the largest gap between eigenvalues
    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors

def silhouette_optimal_k(similarity, k_range, plot=True):
    """
    Automatically select the optimal number of clusters using Silhouette score.
    
    Parameters:
    -----------
    similarity : array-like
        Similarity matrix
    k_range : list or range
        Range of possible cluster numbers to evaluate
    plot : bool, default=True
        Whether to plot the Silhouette scores
        
    Returns:
    --------
    best_k : int
        Optimal number of clusters
    """
    best_k = None
    best_score = -1
    print(f'Silhouette score computation:')
    
    scores_record = []
    for k in k_range:
        # Perform spectral clustering with current k
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed', 
                                      assign_labels='cluster_qr', random_state=42)
        labels = clustering.fit_predict(similarity)
        
        # Convert similarity to dissimilarity for Silhouette score calculation
        disimilarity = 1 - similarity
        np.fill_diagonal(disimilarity, 0)
        score = silhouette_score(disimilarity, labels, metric='precomputed')
        print(f'k = {k}, score = {score}')    
        scores_record.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    print(f'The optimal number of clusters is {best_k}.\n------------------')
    
    if plot:
        plt.title("Records of Silhouette scores")
        plt.scatter(k_range, scores_record)
        plt.grid()
        plt.show()
        
    return best_k

def topoNodeCluster(who, method='spectral', optimal='silhouette'):
    """
    Perform topological clustering on nodes based on their co-occurrence patterns.
    
    Parameters:
    -----------
    who : int
        User ID
    method : str, default='spectral'
        Clustering method ('spectral', 'louvain', 'hdbscan', or 'dbscan')
    optimal : str or int, default='silhouette'
        Method for determining optimal cluster number ('gap', 'silhouette', or specific number)
        
    Returns:
    --------
    labels : array-like
        Cluster labels for each node
    """
    # Load user's visit data and location mappings
    visit_dates = SIRLU.visited_date(who)
    id_coords_mapping = SIRLU.load_id_coords_mapping(who)
    total_loc_number = len(id_coords_mapping)
    
    # Extract week-end dates
    week_end_dates, _ = SIRLU.extract_week_ends(visit_dates)
    
    # Load or compute weekly clustering results
    res_dir = topoResPath(who)
    if not os.path.exists(res_dir):
        cluster_by_week = [clusterLocations(who, week_end_date) for week_end_date in week_end_dates]
    else:
        compute_res = [file for file in os.listdir(res_dir) if file.endswith('.pickle')]
        cluster_by_week = [SIRLU.load_pickle_binary(os.path.join(res_dir, file)) for file in compute_res]
    _, id_coorders_mapping_edit_list, _, cluster_labels_list = list(zip(*cluster_by_week))
    
    # Initialize similarity matrix and appearance counter
    similarity = np.zeros((total_loc_number, total_loc_number))
    appearance = np.zeros(total_loc_number)
    
    # Calculate co-occurrence based similarity
    for id_coorders_mapping_edit, cluster_label in zip(id_coorders_mapping_edit_list, cluster_labels_list):
        location_ids = np.array(list(id_coorders_mapping_edit.keys()))
        appearance[location_ids] += 1
        
        # Create co-occurrence matrix for current week
        cluster_label_row = np.array(cluster_label)
        cluster_label_column = np.array(cluster_label).reshape(-1, 1)
        cluster_label_judge = cluster_label_row == cluster_label_column
        
        # Create upper triangular mask
        seg_loc_len = len(cluster_label)
        masker = np.full((seg_loc_len, seg_loc_len), False)
        masker[np.triu_indices(seg_loc_len, 1)] = True
        
        # Update similarity matrix based on co-occurrences
        co_idx = np.where(cluster_label_judge & masker)
        co_ids = [(location_ids[i], location_ids[j]) for i, j in zip(*co_idx)]
        for co_id in co_ids:
            similarity[co_id] += 1
            
    # Normalize similarity by appearance frequency
    appearance_2d = appearance[:, np.newaxis]
    appearance_base = np.minimum(appearance_2d, appearance_2d.T)
    similarity_corrected = similarity / appearance_base
    
    # Create symmetric affinity matrix
    affinity = similarity_corrected + similarity_corrected.T
    affinity[np.where(affinity == 0)] += 1e-6
    np.fill_diagonal(affinity, 0)
    
    # Perform clustering based on specified method
    if method == 'spectral':
        if optimal == 'gap':
            n_clusters, *_ = eigenDecomposition(affinity, plot=True)
        elif optimal == 'silhouette':
            start_number, end_number = floor(0.1 * total_loc_number), floor(0.9 * total_loc_number)
            n_clusters = silhouette_optimal_k(affinity, range(start_number, end_number + 1))
        elif isinstance(optimal, int):
            n_clusters = optimal
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                     eigen_solver='arpack', assign_labels='cluster_qr',
                                     random_state=42)
        labels = clusterer.fit_predict(affinity)
    elif method == 'louvain':
        G = nx.from_numpy_matrix(affinity)
        communes = louvain_communities(G)
        communes = list(communes)
        
        labels = np.full(len(affinity), -1)
        for i, within_same_communes in enumerate(communes):
            cluster_idx = np.array(list(within_same_communes))
            labels[cluster_idx] = i
    elif method == 'hdbscan':
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=2, metric='precomputed')
        disimilarity = 1 - affinity
        np.fill_diagonal(disimilarity, 0)
        labels = clusterer.fit_predict(disimilarity)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=1/3, min_samples=2, metric='precomputed')
        disimilarity = 1 - affinity
        np.fill_diagonal(disimilarity, 0)
        labels = clusterer.fit_predict(disimilarity)
    return labels


def nodeVerTraj(who, labels):
    """
    Convert location IDs in trajectories to their corresponding cluster labels.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_chain_list : list
        List of trajectories where locations are replaced by their cluster labels
    """
    visit_dates = SIRLU.visited_date(who)
    traj_total = SIRLU.load_all_traj(who)
    
    # Map location IDs to cluster labels
    id_chain_list = [traj.id_chain for traj in traj_total]
    id_node_map = {i: lab for i, lab in enumerate(labels)}
    node_chain_list = [[id_node_map[iden] for iden in id_chain] for id_chain in id_chain_list]
    assert len(visit_dates) == len(node_chain_list), "The date sequence does not match chain sequence."
    
    return node_chain_list

def nodeVisitScan(who, labels):
    """
    Create a binary matrix indicating presence/absence of each cluster in each trajectory.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_scan : list of bool arrays
        Binary matrix where each row represents a cluster and each column represents a trajectory
    """
    node_num = np.max(labels)
    node_chain_list = nodeVerTraj(who, labels)
    node_scan = [[node in node_chain for node_chain in node_chain_list] for node in range(node_num)]
    return node_scan

def nodeTrajCount(who, labels):
    """
    Count the frequency of each unique trajectory pattern in terms of cluster sequences.
    
    Parameters:
    -----------
    who : int
        User ID
    labels : array-like
        Cluster labels for each location
        
    Returns:
    --------
    node_count : Counter
        Counter object containing frequencies of each trajectory pattern
    """
    node_chain_list = nodeVerTraj(who, labels)
    node_chain_tuple = [tuple(l) for l in node_chain_list]
    node_count = Counter(node_chain_list)
    return node_count


def patternRewardChange(node_trajs, large, small):
    assert large > small, "large must be larger than small"
    trajs_large = [tuple(traj) for traj in node_trajs if len(traj) == large]
    trajs_small = [tuple(sorted(traj)) for traj in node_trajs if len(traj) == small]
    trajs_large_res = []; trajs_small_res = []
    for traj in trajs_large:
        # tuple_traj长度为large, 选择其中长度为small的组合
        comb_set = set(combinations(traj, small))
        for comb in comb_set:
            comb_sorted = tuple(sorted(comb))
            if comb_sorted in trajs_small:
                trajs_large_res.append(traj)
                trajs_small_res.append(comb_sorted)
    return trajs_large_res, trajs_small_res


def patternQValueChange(node_trajs, seq_len):
    '''requirements: the node_trajs must be unique'''
    trajs = [tuple(traj) for traj in node_trajs if len(traj) == seq_len]
    recorder = defaultdict(list)
    for traj in trajs:
        ordered_bag = tuple(sorted(traj))
        recorder[ordered_bag].append(traj)
    counter = {order: len(lst_traj) for order, lst_traj in recorder.items()}
    result = {order: lst_traj for order, lst_traj in recorder.items() if counter[order] > 1}
    return result


def JensonShannonDistance(mu1, mu2, sigma1, sigma2):
    '''
    mu1, mu2: mean of two distributions
    sigma1, sigma2: standard deviation of two distributions
    '''
    return 0.5 * (mu1 - mu2)**2 + 0.5 * (sigma1 - sigma2)**2


def nodeFeatureTable(who, id2node):
    features_query = SIRLU.load_state_attrs(who)
    features_query.set_index('fnid', inplace=True)
    id_coords_mapping = SIRLU.load_id_coords_mapping(who)
    coords_fnid_mapping = SIRLU.load_fnid_coords_mapping(who)
    node2ids = Eval.group_locations_by_node(id2node)
    # drop the -1 node from the node2ids dictionary
    if -1 in node2ids:
        node2ids.pop(-1)
    
    id_total_number = len(id_coords_mapping)
    features_table = [[] for _ in range(id_total_number)]
    for id, coords in id_coords_mapping.items():
        fnid = coords_fnid_mapping[coords]
        feature_be_id = features_query.loc[fnid, :].to_numpy()
        coords_id = np.array(coords)
        features_table[id] = np.concatenate([feature_be_id, coords_id])
    
    features_array = np.array(features_table)
    features_by_node_array = np.zeros((len(node2ids), features_array.shape[1]))
    for node, ids in node2ids.items():
        features_by_node_array[node, :] = np.mean(features_array[ids], axis=0)
    
    return features_by_node_array


def nodePiValueTransform(QValue, id2node):
    QValue = np.array(QValue)
    node2ids = Eval.group_locations_by_node(id2node)
    # drop the -1 node from the node2ids dictionary
    if -1 in node2ids:
        node2ids.pop(-1)
    shape0, shape1, _ = QValue.shape
    QNodeValue = np.zeros((shape0, shape1, len(node2ids)))
    for node, ids in node2ids.items():
        QNodeValue[..., node] = np.mean(QValue[..., ids], axis=-1)
    node_probs = softmax(QNodeValue, axis=-1)
    return node_probs
    

def calculateRewardChange(who, id2node, pattern, model_dir):
    node_feature_dataset = nodeFeatureTable(who, id2node)
    node_bag_list, node_sub_bag_list = pattern
    
    # load the model
    data_dir = SIRLU.UserDataPart + '{:09d}/'.format(who)
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    inputs, targets_action, positions, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    mapping_dict = SIRLU.create_coords_utm_mapping(who)
    model = SIRLT.avril(inputs, targets_action, positions, state_dim, action_dim, state_only=True, coords_proj=mapping_dict)
    model_param_dir = os.path.join(model_dir, SIRLU.toWhoString(who), 'evolution_model')
    model_name = sorted(os.listdir(model_param_dir))[-1]
    model_param_path = os.path.join(model_param_dir, model_name)
    model.loadParams(model_param_path)
    
    js_dist_record = []
    dstrbn_record = []
    for node_bag, node_sub_bag in zip(node_bag_list, node_sub_bag_list):
        check_nodes = tuple(set(list(node_sub_bag)))
        
        node_bag_index = list(node_bag)
        node_sub_bag_index = list(node_sub_bag)
        
        inputs_original = node_feature_dataset[None, node_bag_index, None, :-2]
        inputs_shrinked = node_feature_dataset[None, node_sub_bag_index, None, :-2]
        positions_original = node_feature_dataset[None, node_bag_index, None, -2:]
        positions_shrinked = node_feature_dataset[None, node_sub_bag_index, None, -2:]
        
        check_node_index = jnp.array([node_bag_index.index(n) for n in check_nodes])
        check_sub_node_index = jnp.array([node_sub_bag_index.index(n) for n in check_nodes])
        
        reward_original = model.reward(inputs_original, positions_original).squeeze(axis=0)
        reward_shrinked = model.reward(inputs_shrinked, positions_shrinked).squeeze(axis=0)
        
        reward_tgt_original = reward_original[check_node_index, :]
        reward_tgt_shrinked = reward_shrinked[check_sub_node_index, :]
        
        for i in range(len(reward_tgt_original)):
            mu1, log_sigma1 = reward_tgt_original[i, :]
            mu2, log_sigma2 = reward_tgt_shrinked[i, :]
            sigma1 = jnp.exp(log_sigma1)
            sigma2 = jnp.exp(log_sigma2)
            js_dist = js_divergence_normal(mu1, sigma1, mu2, sigma2)
            js_dist_record.append(js_dist)
            dstrbn_record.append(np.array([mu1, sigma1, mu2, sigma2]))
    
    return js_dist_record, dstrbn_record
    
    
def calculateQValueChange(who, id2node, pattern, model_dir):
    node_feature_dataset = nodeFeatureTable(who, id2node)
        
    # load the model
    data_dir = SIRLU.UserDataPart + '{:09d}/'.format(who)
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    inputs, targets_action, positions, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    mapping_dict = SIRLU.create_coords_utm_mapping(who)
    model = SIRLT.avril(inputs, targets_action, positions, state_dim, action_dim, state_only=True, coords_proj=mapping_dict)
    model_param_dir = os.path.join(model_dir, SIRLU.toWhoString(who), 'evolution_model')
    model_name = sorted(os.listdir(model_param_dir))[-1]
    model_param_path = os.path.join(model_param_dir, model_name)
    model.loadParams(model_param_path)

    js_policy_total = []
    distribute_total = []
    for node_bag, node_seqs in pattern.items():
        node_seq_as_indices = np.array(node_seqs)
        num_of_node_seqs, timespan = node_seq_as_indices.shape
        features = node_feature_dataset[node_seq_as_indices]
        inputs_features = features[:, :, None, :-2]
        positions_features = features[:, :, None, -2:]
        QValues = model.QValue(inputs_features, positions_features)
        PiNodeValues = nodePiValueTransform(QValues, id2node)
                
        for i in range(num_of_node_seqs):
            for j in range(i + 1, num_of_node_seqs):
                seq_base = node_seqs[i]
                seq_comp = node_seqs[j]
                first_index = np.where(np.array(seq_base) != np.array(seq_comp))[0][0]
                # 获取first_index之后的子序列
                seq_comp_after = list(seq_comp[first_index:])
                for k in range(first_index, timespan):
                    k_th_node = seq_base[k]
                    temp_idx = seq_comp_after.index(k_th_node)
                    k_mirror = first_index + temp_idx
                    seq_comp_after[temp_idx] = -1
                    p1, p2 = PiNodeValues[i, k, :], PiNodeValues[j, k_mirror, :]
                    js_policy_total.append(js_distance_discrete(p1, p2)) 
                    distribute_total.append((p1, p2))
    return js_policy_total, distribute_total


def js_distance_discrete(p, q, base=2):
    """
    计算两个离散型概率分布的Jensen-Shannon距离（即JSD的平方根）

    参数:
    ----
    p, q : array-like
        两个概率分布（元素非负，且和为1）
    base : float, optional
        对数的底，默认为2（与信息熵一致）

    返回:
    ----
    js_dist : float
        Jensen-Shannon距离
    """
    # 转为numpy数组
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)
    # 计算JS距离
    js_dist = jensenshannon(p, q, base=base)
    return js_dist


def js_divergence_normal(mu1, sigma1, mu2, sigma2, x_range=None, n_points=1000):
    """
    计算两个正态分布的Jensen-Shannon散度
    
    Parameters:
    -----------
    mu1, mu2 : float
        两个正态分布的均值
    sigma1, sigma2 : float
        两个正态分布的标准差
    x_range : tuple, optional
        计算范围 (min_x, max_x)，如果为None则自动计算
    n_points : int, default=1000
        离散化点数
        
    Returns:
    --------
    js_div : float
        Jensen-Shannon散度值
    """
    # 如果未指定范围，自动计算合适的范围
    if x_range is None:
        # 覆盖两个分布的99.7%范围（3个标准差）
        min_x = min(mu1 - 3*sigma1, mu2 - 3*sigma2)
        max_x = max(mu1 + 3*sigma1, mu2 + 3*sigma2)
        x_range = (min_x, max_x)
    
    # 创建离散化的x轴
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # 计算两个正态分布的PDF
    p1 = norm.pdf(x, mu1, sigma1)
    p2 = norm.pdf(x, mu2, sigma2)
    
    # 归一化概率分布
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    
    # 计算JS散度
    js_div = jensenshannon(p1, p2)
    
    return js_div



if __name__ == "__main__":
    who = 58124481
    cluster_res = topoNodeCluster(who)
    seq_res = nodeVisitScan(who, cluster_res)
    print(seq_res)
    
    # 测试JS散度函数
    print("\n测试JS散度计算:")
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 2, 1.5
    js_num = js_divergence_normal(mu1, sigma1, mu2, sigma2)
    print(f"数值方法JS散度: {js_num:.6f}")
    