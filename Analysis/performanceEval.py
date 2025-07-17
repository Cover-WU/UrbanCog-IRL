import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

# 需要用到的包
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
from SCBIRL_Global_PE.utils import Padding
from Analysis.topoMap import clusterLocationsSimple

MODEL_DIR = './model/'
USER_DATA_DIR = SIRLU.UserDataPart

# ========== 工具函数 ==========
def get_evolution_model_paths(who):
    """返回该用户所有evolution_model的路径和日期列表"""
    who_str = SIRLU.toWhoString(who)
    model_dir = os.path.join(MODEL_DIR, who_str, 'evolution_model')
    if not os.path.exists(model_dir):
        return []
    model_files = [f for f in os.listdir(model_dir) if f.startswith('iterated_model_') and f.endswith('.pickle')]
    # 提取日期
    date_list = [int(f.split('_')[-1].replace('.pickle','')) for f in model_files]
    model_paths = [os.path.join(model_dir, f) for f in model_files]
    # 按日期排序
    date_model = sorted(zip(date_list, model_paths))
    return date_model

def get_next_day_traj(who, date):
    """获取date之后一天的所有轨迹（list of TravelData）"""
    all_traj = SIRLU.load_all_traj(who)
    # 找到第一个date等于给定date的tc的索引
    idx_list = [i for i, tc in enumerate(all_traj) if tc.date == date]
    if not idx_list:
        # 报告一个warning
        print(f"Warning: No date {date} for user {who}.")
        return []
    elif idx_list[0] + 1 == len(all_traj): 
        return []
    else:
        return [all_traj[idx_list[0] + 1]]

def get_this_day_traj(who, date):
    """获取date当天所有轨迹（list of TravelData）"""
    all_traj = SIRLU.load_all_traj(who)
    return [tc for tc in all_traj if tc.date == date]

def load_id_node_maparray(who):
    """返回id到node的映射（聚类标签）"""
    labels = clusterLocationsSimple(who)
    return labels

def traj_id2node(traj, id2node):
    '''input one trajectory'''
    # 检查trajectory不能是列表
    if isinstance(traj, list):
        raise ValueError("Input must be a single trajectory, not a list.")
    return [id2node[id] for id in traj.id_chain]

def group_locations_by_node(id2node):
    """返回node->location id的映射dict"""
    node2ids = defaultdict(list)
    for id, node in enumerate(id2node):
        node2ids[node].append(id)
    return node2ids

def traj_processor(who, trajs):
    """处理trajectory，返回输入格式"""
    state_attribute = SIRLU.load_state_attrs(who)
    s_dim = state_attribute.shape[1] - 1
    states, actions, positions = SIRLU.processTrajectoryData(trajs, state_attribute, s_dim)    
    return states, actions, positions

# ========== 评估主流程 ==========

def evaluate_user_model(model, who, id2node, node2ids, next_trajs, today_trajs):
    
    next_trajs_input = traj_processor(who, next_trajs)
    today_trajs_input = traj_processor(who, today_trajs)

    acc1, acc5, step_n  = eval_topk_accuracy(model, next_trajs_input, id2node, node2ids, k=5)
    # 2. Perplexity
    perplex, perplex_n = eval_perplexity(model, today_trajs_input, id2node, node2ids)
    # 3. BLEU/ROUGE
    bleu, rouge, bleu_n, rouge_n = eval_bleu_rouge(model, next_trajs_input, id2node, node2ids)
    # 4. 编辑距离
    edit_dist, edit_n = eval_edit_distance(model, next_trajs_input, id2node, node2ids)
    return {
        'acc1': acc1, 'acc5': acc5, 'step_n': step_n,
        'perplex': perplex, 'perplex_n': perplex_n,
        'bleu': bleu, 'bleu_n': bleu_n, 'rouge': rouge, 'rouge_n': rouge_n,
        'edit_dist': edit_dist, 'edit_n': edit_n
    }


def evaluate_user(who):
    """对单个用户所有evolution_model做评估，返回结果dict"""
    # 最后可以转换为嵌套列表，然后转换为DataFrame，便于统计
    id2node = load_id_node_maparray(who)
    node2ids = group_locations_by_node(id2node)
    one_usr_results = dict()
    date_model = get_evolution_model_paths(who)
    for date, model_path in tqdm(date_model, desc=f"User {who}"):
        # 加载模型
        model = SIRLT.avril(...)
        model.loadParams(model_path)
        next_trajs = get_next_day_traj(who, date)
        today_trajs = get_this_day_traj(who, date)
        eval_results = evaluate_user_model(model, who, id2node, node2ids, next_trajs, today_trajs)
        one_usr_results[date] = eval_results
    return one_usr_results

# ========== 评估指标实现（伪代码/接口） ==========
def eval_topk_accuracy(model, trajs_input, id2node, node2ids, shuffle=True, k=5):
    """
    单步Top-1/5准确率，返回acc1, acc5, 步数。
    支持对轨迹序列打乱。
    """
    acc1_hits = 0
    acc5_hits = 0
    total_steps = 0
    
    states, actions, positions = trajs_input
    for traj_idx in range(len(states)):
        # 正常情况下，trajs本来就是一条轨迹，所以traj_idx=0
        state = states[traj_idx, :, 0, np.newaxis, :]
        value_mask = ~np.all(state == Padding, axis=0)
        state = state[value_mask]
        action = actions[traj_idx, :, 0, np.newaxis, :]
        action = action[value_mask]
        position = positions[traj_idx, :, 0, np.newaxis, :]
        position = position[value_mask]
        time_span = len(state)
        # 至少要有两步才能做预测
        if time_span < 2:
            continue
        for t in range(time_span - 1):
            next_id = action[t+1]
            next_node = id2node[next_id]
            # 构造当前状态和位置
            # 你可能需要根据你的模型输入格式调整
            state_encode, position_encode = state.copy(), position.copy()
            if shuffle:
                np.random.shuffle(state_encode)
                np.random.shuffle(position_encode)
            
            state_encode_tensor, state_tensor, position_encode_tensor, position_tensor =    \
                state_encode[np.newaxis, :, np.newaxis, :], state[np.newaxis, :, np.newaxis, :],   \
                position_encode[np.newaxis, :, np.newaxis, :], position[np.newaxis, :, np.newaxis, :]
            QArray = model.inference_rollout_QValue(state_encode_tensor, position_encode_tensor, state_tensor, position_tensor)
            QArray = QArray.squeeze()
            # 对每个node内的location做平均
            nodeQ = {}
            for node, ids in node2ids.items():
                nodeQ[node] = np.mean([QArray[loc] for loc in ids])
            # softmax
            nodeQ_arr = np.array([nodeQ[n] for n in sorted(nodeQ.keys())])
            node_probs = np.exp(nodeQ_arr - np.max(nodeQ_arr))
            node_probs = node_probs / np.sum(node_probs)
            # Top-1/Top-5
            topk_idx = np.argsort(node_probs)[::-1][:k]
            topk_nodes = [sorted(nodeQ.keys())[i] for i in topk_idx]
            top1_node = topk_nodes[0]
            # 真实下一步node - 从action[t+1]获取，而不是traj.id_chain
            next_id = int(action[t+1].item())  # 确保是标量
            next_node = id2node[next_id]
            # 命中统计
            if next_node == top1_node:
                acc1_hits += 1
            if next_node in topk_nodes:
                acc5_hits += 1
            total_steps += 1
    acc1 = acc1_hits / total_steps if total_steps > 0 else None
    acc5 = acc5_hits / total_steps if total_steps > 0 else None
    if total_steps == 0:
        total_steps = None
    return acc1, acc5, total_steps

def eval_perplexity(model, trajs, id2node, node2ids):
    """困惑度，返回perplex, 步数"""
    # TODO: 实现
    return 0, 0

def eval_bleu_rouge(model, trajs, id2node, node2ids):
    """BLEU/ROUGE，返回bleu, rouge, 轨迹数"""
    # TODO: 实现
    return 0, 0, 0, 0

def eval_edit_distance(model, trajs, id2node, node2ids):
    """编辑距离，返回归一化编辑距离，轨迹数"""
    # TODO: 实现
    return 0, 0

# ========== 主入口 ==========
def main():
    user_list = [int(name) for name in os.listdir(MODEL_DIR) if name.isdigit()]
    all_results = []
    for who in user_list:
        res = evaluate_user(who)
        all_results.extend(res)
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_csv('./Analysis/performance_eval_results.csv', index=False)
    with open('./product/performance_eval_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print('Evaluation finished.')

if __name__ == '__main__':
    main() 