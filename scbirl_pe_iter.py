import pickle
import copy
import os
import SCBIRL_Global_PE.SCBIRLTransformer as SIRLT
import SCBIRL_Global_PE.utils as SIRLU
import SCBIRL_Global_PE.migrationProcess as SIRLP
import Analysis.priorKnow as PriorKnow
from SCBIRL_Global_PE.utils import Traveler, UserDataPart
# from Analysis.comparison_models import avril_without_pe

import jax
jax.config.update('jax_platform_name', 'cpu')

def train_model_one_traveler(who: int):
    data_dir = UserDataPart + '{:09d}/'.format(who)
    model_dir = './model/{:09d}/'.format(who)
    os.makedirs(model_dir, exist_ok=True)
    
    iter_start_date = SIRLU.load_traveler(who).iter_start_date
    # here the `iter_start_date` is a constant defined by utility module.
    inputs, targets_action, positions, pe_code, action_dim, state_dim = SIRLU.loadTrajChain(data_dir, type='before', start_date=iter_start_date)
    print(inputs.shape, targets_action.shape, positions.shape, pe_code.shape)
    # tabular rasa model
    model = SIRLT.avril(inputs, targets_action, positions, pe_code, state_dim, action_dim, state_only=True)
    # model = avril_without_pe(inputs, targets_action,  state_dim, action_dim, state_only=True)

    # # model the model with no prior knowledge
    # PriorKnow.experienceModel(model, data_dir, model_dir, start_date = iter_start_date)
    # # NOTE: Compute rewards after migration
    # model_no_prior = copy.deepcopy(model)
    # SIRLP.afterMigrt(model_no_prior, data_dir, model_dir, start_date = iter_start_date, iter_type='prior')

    # NOTE: train the model before migration
    model.train(iters=1000, loss_threshold=0.01)
    model_repo_establishment(model, model_dir)
    model_save_path = model_dir + 'initial_model.pickle'
    model.modelSave(model_save_path)

    # NOTE: Compute rewards after migration
    SIRLP.afterMigrt(model, data_dir, model_dir, start_date = iter_start_date, iter_type='recent')

def model_repo_establishment(model: SIRLT.avril, path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    # create a txt file to record the model configuration
    num_layers = model.num_layers
    num_heads = model.num_heads
    num_scales = model.num_scale
    dff_ratio = model.dff
    dropout_rate = model.rate
    
    text = f"""
    Model Configuration:
    -------------------
    Number of layers: {num_layers}
    Number of heads: {num_heads}
    Number of scales: {num_scales}
    Feedforward ratio: {dff_ratio}
    Dropout rate: {dropout_rate}
    """
    with open(path + 'model_config.txt', 'w') as f:
        f.write(text)
        
if __name__ =="__main__":
    '''
        Iteration Version
    '''
    # for who in who_list:
    #     train_model_one_traveler(who = who)

    '''
        Parallel Version
    '''
    # import multiprocessing as mp
    
    # MAX_CPU_COUNT = mp.cpu_count() - 2
    # done_who = []
    # import os
    # file_list = os.listdir(UserDataPart)
    # who_list = [int(pid) for pid in file_list]
    # who_list = [10013454]
    # for who in done_who:
    #     who_list.remove(who)
    # with mp.Pool(MAX_CPU_COUNT) as pool:
    #     pool.map(train_model_one_traveler, who_list)

    '''
        Terminal Version
    '''
    train_model_one_traveler(who = 82455786)
