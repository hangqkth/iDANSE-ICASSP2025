#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################
import numpy as np
import torch
import sys
import os
import copy
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.danse import DANSE
# from src.danseq import DANSeq, IDANSE
from src.idanse import IDANSE
from parameters_opt import get_parameters
from torch.autograd import Variable
from utils.utils import push_model

def test_danse_ssm(Y_test, ssm_model_test, model_file_saved_danse, Cw_test=None, rnn_type='gru', device='cpu', bidrectional=False):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch

    danse_model = DANSE(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        batch_size=1,
        H=ssm_model_test.H,
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
        )
    
    print("DANSE Model file: {}".format(model_file_saved_danse))

    start_time_danse = timer()
    danse_model.load_state_dict(torch.load(model_file_saved_danse, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch, 
                                                                                                                           Cw_test_batch)
    
    time_elapsed_danse = timer() - start_time_danse

    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered, time_elapsed_danse


def test_idanse_ssm(Y_test, ssm_model_test, model_file_saved_idanse_list, Cw_test=None, rnn_type='gru', device='cpu'):
    # Initialize the DANSE model parameters
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs,
                                        device=device)




    # Initialize the DANSE model in PyTorch
    idanse_model = IDANSE(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        batch_size=1,
        H=ssm_model_test.H,
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )
    # Build iteration with k times, k is depending on the length of model_file_saved_idanse_list
    model_file_saved_idanse_list = model_file_saved_idanse_list[1:]
    k_list = []
    for i in range(len(model_file_saved_idanse_list)):
        k_num = model_file_saved_idanse_list[i].split('_k_')[-1].split('_')[0]
        k_list.append(int(k_num))
    k_list_sorted = sorted(k_list)
    file_list_sorted = []
    for k_item in k_list_sorted:
        loc = k_list.index(k_item)
        file_list_sorted.append(model_file_saved_idanse_list[loc])

    start_time_danse = timer()

    # add a loop here
    for k in range(len(file_list_sorted)):
        # idanse_model.load_state_dict(torch.load(file_list_sorted[k], map_location=device))
        idanse_model.load_state_dict(torch.load(file_list_sorted[-1], map_location=device))
        idanse_model = push_model(nets=idanse_model, device=device)
        idanse_model.eval()

        with torch.no_grad():
            Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
            Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
            if k == 0:
                x_hat_test = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
            X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = idanse_model.compute_predictions(
                Y_test_batch, x_hat_test, Cw_test_batch)
            x_hat_test = X_estimated_filtered

    time_elapsed_danse = timer() - start_time_danse

    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered, time_elapsed_danse



if __name__ == "__main__":

    model_file_saved_idanse_list = ['../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_BEST.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_1_best.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_23_best.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_2_best.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_30_best.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_4_best.pt',
     '../run/models/LorenzSSM_danse_opt_gru_m_3_n_3_T_100_N_1000_sigmae2_-10.0dB_smnr_-10.0dB/idanse_gru_ckpt_k_6_best.pt']

    # Build iteration with k times, k is depending on the length of model_file_saved_idanse_list
    model_file_saved_idanse_list = model_file_saved_idanse_list[1:]
    k_list = []
    for i in range(len(model_file_saved_idanse_list)):
        k_num = model_file_saved_idanse_list[i].split('_k_')[-1].split('_')[0]
        k_list.append(int(k_num))
    k_list_sorted = sorted(k_list)
    file_list_sorted = []
    for k_item in k_list_sorted:
        loc = k_list.index(k_item)
        file_list_sorted.append(model_file_saved_idanse_list[loc])




