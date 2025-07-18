import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
from scipy.special import softmax

# 需要用到的包
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import editdistance

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

def load_id_node_mapping(who):
    """返回id到node的映射（聚类标签）"""
    labels = clusterLocationsSimple(who)
    # 在末尾增加一个元素为-1
    mapping = {i: label for i, label in enumerate(labels)}
    mapping[-1] = -1
    return mapping

def traj_id2node(trajs, id2node):
    '''input one trajectory'''
    # 检查trajectory不能是列表
    if isinstance(trajs, list):
        return [[id2node[id] for id in traj.id_chain] for traj in trajs]
    else:
        return [id2node[id] for id in trajs.id_chain]

def group_locations_by_node(id2node):
    """返回node->location id的映射dict"""
    node2ids = defaultdict(list)
    for id, node in id2node.items():
        node2ids[node].append(id)
    return node2ids

def traj_processor(who, trajs):
    """处理trajectory，返回输入格式"""
    state_attribute = SIRLU.load_state_attrs(who)
    s_dim = state_attribute.shape[1] - 1
    states, actions, positions = SIRLU.processTrajectoryData(trajs, state_attribute, s_dim) 
    
    traj_inputs = []
    for i in range(len(trajs)):
        state = states[i, :, 0, :]
        value_mask = ~np.all(state == Padding, axis=0)
        state = state[value_mask]
        action = actions[i, :, 0, :]
        action = action[value_mask]
        position = positions[i, :, 0, :]
        position = position[value_mask]
        
        input_per_traj = (state, action, position)
        traj_inputs.append(input_per_traj)
    
    return traj_inputs

# ========== 评估主流程 ==========

def evaluate_user_model(model, who, id2node, node2ids, next_trajs, today_trajs):
    
    next_trajs_input = traj_processor(who, next_trajs)
    today_trajs_input = traj_processor(who, today_trajs)

    acc1, acc5, step_n  = eval_topk_accuracy(model, next_trajs_input, id2node, node2ids, k=5)
    # 2. Perplexity
    perplex, perplex_n = eval_perplexity(model, next_trajs_input, id2node, node2ids)
    # 3. BLEU/ROUGE
    bleu, rouge, timespan, n_traj = eval_bleu_rouge(model, who, today_trajs, id2node, node2ids)
    # 4. 编辑距离
    edit_dist, edit_n = eval_edit_distance(model, who, today_trajs, id2node, node2ids)
    return {
        'acc1': acc1, 'acc5': acc5, 'step_n': step_n,
        'perplex': perplex, 'perplex_n': perplex_n,
        'bleu': bleu, 'rouge': rouge, 'timespan': timespan, 'n_traj': n_traj,
        'edit_dist': edit_dist, 'edit_n': edit_n
    }


def evaluate_user(who):
    """对单个用户所有evolution_model做评估，返回结果dict"""
    # 最后可以转换为嵌套列表，然后转换为DataFrame，便于统计
    id2node = load_id_node_mapping(who)
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
    acck_hits = 0
    total_steps = 0

    for traj_one_day in trajs_input:
        # 正常情况下，trajs本来就是一条轨迹，所以traj_idx=0
        state, action, position = traj_one_day
        time_span = len(state)
        # 至少要有两步才能做预测
        # 构造当前状态和位置
        # 你可能需要根据你的模型输入格式调整
        state_encode, position_encode = state.copy(), position.copy()
        if shuffle:
            np.random.shuffle(state_encode)
            np.random.shuffle(position_encode)

        state_encode_tensor, state_tensor, position_encode_tensor, position_tensor =    \
            state_encode[None, :, None, :], state[None, :, None, :],   \
            position_encode[None, :, None, :], position[None, :, None, :]
        QArray = model.inference_rollout_QValue(state_encode_tensor, position_encode_tensor, state_tensor, position_tensor)
        QArray = QArray.squeeze(axis=(0, 2))
        # 对每个node内的location做平均
        QNodeArray = np.zeros((len(QArray), len(node2ids)))
        for node, ids in node2ids.items():
            # TODO 这里注意一下遍历性，不要遗漏
            QNodeArray[:, node] = np.mean(QArray[:, ids], axis=1)
            # 如果模型能自动切割分配动作值，也可以求和，我估计模型没有这个能力，最大熵要求满足IIA假设。这个似乎不满足的。
            # QNodeArray[:, node] = np.sum(QArray[:, ids], axis=1)
        # softmax
        node_probs = softmax(QNodeArray, axis=1)
        # 选择每一行排名前k的node index
        topk_nodes = np.argsort(node_probs)[:, ::-1][:, :k]

        next_id = action[:, 0]
        next_node = np.array([id2node[int(nid)] for nid in next_id])
        top1_node = topk_nodes[:, 0]
        # check the top1 accuracy number
        top1_correct = np.sum(top1_node == next_node)
        # check the topk accuracy number
        topk_correct = np.any(topk_nodes == next_node[:, None], axis=1).sum()
        
        acc1_hits += top1_correct
        acck_hits += topk_correct
        total_steps += time_span
    
    acc1 = acc1_hits / total_steps if total_steps > 0 else None
    acck = acck_hits / total_steps if total_steps > 0 else None
    if total_steps == 0:
        total_steps = None
    
    return acc1, acck, total_steps

def eval_perplexity(model, trajs_input, id2node, node2ids, shuffle=False):
    """
    计算困惑度（perplexity），返回perplex, 步数。
    """
    log_prob_sum = 0.0
    total_steps = 0

    # node_list = sorted(node2ids.keys())
    for traj_one_day in trajs_input:
        state, action, position = traj_one_day
        time_span = len(state)
        state_encode, position_encode = state.copy(), position.copy()
        if shuffle:
            np.random.shuffle(state_encode)
            np.random.shuffle(position_encode)
        state_encode_tensor = state_encode[None, :, None, :]
        state_tensor = state[None, :, None, :]
        position_encode_tensor = position_encode[None, :, None, :]
        position_tensor = position[None, :, None, :]
        QArray = model.inference_rollout_QValue(state_encode_tensor, position_encode_tensor, state_tensor, position_tensor)
        QArray = QArray.squeeze(axis=(0, 2))
        QNodeArray = np.zeros((len(QArray), len(node2ids)))
        for node, ids in node2ids.items():
            QNodeArray[:, node] = np.mean(QArray[:, ids], axis=1)
        
        node_probs = softmax(QNodeArray, axis=1)
        # 真实下一步node
        next_id = action[:, 0]
        next_node = [id2node[int(nid)] for nid in next_id]
        
        prob_seq = node_probs[np.arange(len(next_node)), next_node]
        log_prob_seq = np.log(prob_seq)
        log_prob_sum += np.sum(log_prob_seq)
        total_steps += time_span
        
    perplex = np.exp(-log_prob_sum / total_steps) if total_steps > 0 else None
    if total_steps == 0:
        total_steps = None
    return perplex, total_steps


def eval_bleu_rouge(model, who, trajs_input, id2node, node2ids, max_gen_len=None, shuffle=False):
    """
    计算BLEU/ROUGE，返回bleu, rouge, 轨迹数
    """
    bleu_scores = []
    rouge_scores = []
    traj_timespans = []
    n_traj = 0

    rouge = Rouge()
    for traj_one_day in trajs_input:
        state, action, position = traj_one_day
        time_span = len(state)
        if time_span < 2:
            continue

        # 真实轨迹的node序列
        true_node_seq = [id2node[int(nid)] for nid in action[:, 0]]

        state_encode, position_encode = state.copy(), position.copy()        
        if shuffle:
            np.random.shuffle(state_encode)
            np.random.shuffle(position_encode)
            
        state_encode_tensor = state_encode[None, :, None, :]
        position_encode_tensor = position_encode[None, :, None, :]
        # 用模型生成轨迹（贪心/采样，node级）
        # 这里用贪心策略：每一步选概率最大的node
        gen_node_seq = []
        curr_state = state[0:1]  # 初始状态 shape (1, state_dim)
        curr_position = position[0:1] # 初始位置 shape (1, 2)
        if max_gen_len is not None:
            time_span = max_gen_len
        
        for t in range(time_span):
            # 构造输入
            state_tensor = curr_state[None, :, None, :]
            pos_tensor = curr_position[None, :, None, :]
        
            QArray = model.inference_rollout_QValue(state_encode_tensor, position_encode_tensor, state_tensor, pos_tensor)
            QArray = QArray.squeeze(axis=(0, 2))
            QNodeArray = np.zeros((len(QArray), len(node2ids)))
            for node, ids in node2ids.items():
                QNodeArray[:, node] = np.mean(QArray[:, ids], axis=1)
            
            node_probs = softmax(QNodeArray, axis=1)
            final_decided_node = node_probs[-1, :].argmax()
            
            if final_decided_node == node_probs.shape[1]:
                # 终止轨迹生成
                gen_node_seq.append(-1)
                break
            else:
                gen_node_seq.append(final_decided_node)
                # 终止符
            
            if t < time_span - 1:
                # 更新curr_state/curr_pos（这里用模型预测）
                # 先求出最有可能的动作
                QSelection = QArray[-1, :][node2ids[final_decided_node]].max()
                QSelection_idx = np.where(QArray[-1, :] == QSelection)[0][0]
                
                # turn action to traj and then append to curr_state/curr_pos
                id_coords_mapping = SIRLU.load_id_coords_mapping(who)
                coords = id_coords_mapping[QSelection_idx]
                coords_fnid = SIRLU.load_fnid_coords_mapping(who)
                fnid = coords_fnid[coords]
                state_attrs = SIRLU.load_state_attrs(who)
                append_state = SIRLU.getStateRow(state_attrs, fnid)
                append_position = np.array(coords)
                curr_state = np.concatenate([curr_state, append_state[None, :]], axis=0)
                curr_position = np.concatenate([curr_position, append_position[None, :]], axis=0)
            
        # 对齐长度
        # min_len = min(len(true_node_seq), len(gen_node_seq))
        true_str = ' '.join(map(str, true_node_seq))
        gen_str = ' '.join(map(str, gen_node_seq))

        # BLEU
        bleu = sentence_bleu([true_node_seq], gen_node_seq)
        bleu_scores.append(bleu)
        # ROUGE
        try:
            rouge_score = rouge.get_scores(gen_str, true_str)[0]['rouge-l']['f']
        except Exception:
            rouge_score = 0.0
        rouge_scores.append(rouge_score)
        # traj_timespans.append(time_span)
        n_traj += 1

    bleu_avg = np.mean(bleu_scores) if bleu_scores else None
    rouge_avg = np.mean(rouge_scores) if rouge_scores else None
    n_traj = n_traj if n_traj else None
    return bleu_avg, rouge_avg, n_traj


def eval_edit_distance(model, who, trajs_input, id2node, node2ids, max_gen_len=None, shuffle=False):
    """
    计算BLEU/ROUGE，返回bleu, rouge, 轨迹数
    """
    n_traj = 0
    edit_dist = []
    time_spans = []
    
    for traj_one_day in trajs_input:
        state, action, position = traj_one_day
        time_span = len(state)
        if time_span < 2:
            continue

        # 真实轨迹的node序列
        true_node_seq = [id2node[int(nid)] for nid in action[:, 0]]

        state_encode, position_encode = state.copy(), position.copy()        
        if shuffle:
            np.random.shuffle(state_encode)
            np.random.shuffle(position_encode)
            
        state_encode_tensor = state_encode[None, :, None, :]
        position_encode_tensor = position_encode[None, :, None, :]
        # 用模型生成轨迹（贪心/采样，node级）
        # 这里用贪心策略：每一步选概率最大的node
        gen_node_seq = []
        curr_state = state[0:1]  # 初始状态 shape (1, state_dim)
        curr_position = position[0:1] # 初始位置 shape (1, 2)
        if max_gen_len is not None:
            time_span = max_gen_len
        
        for t in range(time_span):
            # 构造输入
            state_tensor = curr_state[None, :, None, :]
            pos_tensor = curr_position[None, :, None, :]
        
            QArray = model.inference_rollout_QValue(state_encode_tensor, position_encode_tensor, state_tensor, pos_tensor)
            QArray = QArray.squeeze(axis=(0, 2))
            QNodeArray = np.zeros((len(QArray), len(node2ids)))
            for node, ids in node2ids.items():
                QNodeArray[:, node] = np.mean(QArray[:, ids], axis=1)
            
            node_probs = softmax(QNodeArray, axis=1)
            final_decided_node = node_probs[-1, :].argmax()
            
            if final_decided_node == node_probs.shape[1]:
                # 终止轨迹生成
                gen_node_seq.append(-1)
                break
            else:
                gen_node_seq.append(final_decided_node)
                # 终止符
            
            if t < time_span - 1:
                # 更新curr_state/curr_pos（这里用模型预测）
                # 先求出最有可能的动作
                QSelection = QArray[-1, :][node2ids[final_decided_node]].max()
                QSelection_idx = np.where(QArray[-1, :] == QSelection)[0][0]
                
                # turn action to traj and then append to curr_state/curr_pos
                id_coords_mapping = SIRLU.load_id_coords_mapping(who)
                coords = id_coords_mapping[QSelection_idx]
                coords_fnid = SIRLU.load_fnid_coords_mapping(who)
                fnid = coords_fnid[coords]
                state_attrs = SIRLU.load_state_attrs(who)
                append_state = SIRLU.getStateRow(state_attrs, fnid)
                append_position = np.array(coords)
                curr_state = np.concatenate([curr_state, append_state[None, :]], axis=0)
                curr_position = np.concatenate([curr_position, append_position[None, :]], axis=0)
            
        dist = editdistance.eval(true_node_seq, gen_node_seq)
        edit_dist.append(dist / time_span)
        # time_spans.append(time_span)
        n_traj += 1

    edit_dist_avg = np.mean(edit_dist) if edit_dist else None
    n_traj = n_traj if n_traj else None
    
    return edit_dist_avg, n_traj


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