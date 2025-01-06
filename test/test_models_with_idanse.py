#####################################################
# Creator: Hang Qin, Anubhab Ghosh
# Sep 2024
#####################################################
import numpy as np
import glob
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer
import itertools
import json
import tikzplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import dB_to_lin, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder, partial_corrupt, push_model
#from parameters import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
from parameters_opt import get_parameters, f_lorenz, f_lorenz_ukf, f_chen, f_chen_ukf, f_rossler, f_rossler_ukf, \
    J_test, delta_t, delta_t_chen, delta_t_rossler, \
    decimation_factor_lorenz, decimation_factor_chen, decimation_factor_rossler, get_H_DANSE
from bin.generate_data import LorenzSSM, RosslerSSM
from test_ekf_ssm import test_ekf_ssm
from test_ukf_ssm import test_ukf_ssm
from test_ukf_ssm_one_step import test_ukf_ssm_one_step
from test_danse_ssm import test_danse_ssm, test_idanse_ssm

from get_one_step_ahead_lin_meas import get_y_pred_linear_meas

def metric_to_func_map(metric_name):
    """ Function to map from a metric name to the respective function.
    The metric 'time_elapsed' couldn't be included here, as it is calculated inline,
    and hence needs to be manually assigned.  

    Args:
        metric_name (str): Metric name 

    Returns:
        fn : function call that helps to calculate the given metric
    """
    if metric_name == "nmse":
        fn = nmse_loss
    elif metric_name == "nmse_std":
        fn = nmse_loss_std
    elif metric_name == "mse_dB":
        fn = mse_loss_dB
    elif metric_name == "mse_dB_std":
        fn = mse_loss_dB_std
    return fn

def get_f_function(ssm_type):

    if "LorenzSSM" in ssm_type:
        f_fn = f_lorenz
        f_ukf_fn = f_lorenz_ukf
    elif "ChenSSM" in ssm_type:
        f_fn = f_chen
        f_ukf_fn = f_chen_ukf
    elif "RosslerSSM" in ssm_type:
        f_fn = f_rossler
        f_ukf_fn = f_rossler_ukf

    return f_fn, f_ukf_fn

def test_on_ssm_model(device='cpu', learnable_model_files=None, test_data_file=None, 
            test_logfile=None, evaluation_mode='full', metrics_list=None, models_list=None, 
            dirname=None):
    
    model_file_saved_danse = learnable_model_files["danse"] if "danse" in models_list else None
    model_file_saved_idanse_list = learnable_model_files["idanse"] if "idanse" in models_list else None
    model_file_saved_danse_supervised = learnable_model_files["danse_supervised"] if "danse_supervised" in models_list else None
    model_file_saved_danse_semisupervised = learnable_model_files["danse_semisupervised"] if "danse_semisupervised" in models_list else None

    model_file_saved_danse = model_file_saved_danse.replace('\\', '/')
    model_file_saved_idanse_list = [model_file_saved_idanse.replace('\\', '/') for model_file_saved_idanse in model_file_saved_idanse_list]

    print(models_list)
    ssm_type, rnn_type, m, n, T, _, sigma_e2_dB, smnr_dB = parse("{}_danse_opt_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB", model_file_saved_danse.split('/')[-2])
    
    J = 5
    J_test = 5
    #smnr_dB = 20
    decimate=True
    use_Taylor = False 

    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        print('Dataset is not present, creating at {}'.format(test_data_file), file=orig_stdout)
        # My own data generation scheme
        m, n, ssm_type_test, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        #N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test, m))
        Y = torch.zeros((N_test, T_test, n))
        Cw = torch.zeros((N_test, n, n))

        ssm_dict, _ = get_parameters(n_states=m, n_obs=n, device='cpu')
        ssm_model_test_dict = ssm_dict[ssm_type_test]
        f_fn, f_ukf_fn = get_f_function(ssm_type)

        if "LorenzSSM" in ssm_type:
            delta = delta_t # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = delta_t / decimation_factor_lorenz
            ssm_model_test = LorenzSSM(n_states=m, n_obs=n, J=J, delta=delta, 
                                delta_d=delta_d, decimate=decimate, alpha=0.0,
                                H=get_H_DANSE(type_=ssm_type, n_states=m, n_obs=n),
                                mu_e=np.zeros((m,)), mu_w=np.zeros((n,)),
                                use_Taylor=use_Taylor)
            decimation_factor = decimation_factor_lorenz

        elif "ChenSSM" in ssm_type:
            delta = delta_t_chen # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = delta_t_chen / decimation_factor_chen
            ssm_model_test = LorenzSSM(n_states=m, n_obs=n, J=J, delta=delta, 
                                delta_d=delta_d, decimate=decimate, alpha=1.0,
                                H=get_H_DANSE(type_=ssm_type, n_states=m, n_obs=n),
                                mu_e=np.zeros((m,)), mu_w=np.zeros((n,)),
                                use_Taylor=use_Taylor)
            decimation_factor = decimation_factor_chen

        elif "RosslerSSM" in ssm_type:
            delta = delta_t_rossler # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = delta_t_rossler / decimation_factor_rossler
            ssm_model_test = RosslerSSM(n_states=m, n_obs=n, J=J_test, delta=delta, delta_d=delta_d, 
                                   a=ssm_model_test_dict['a'], b=ssm_model_test_dict['b'],
                                   H=get_H_DANSE(type_=ssm_type, n_states=m, n_obs=n),
                                   c=ssm_model_test_dict['c'], decimate=decimate, mu_e=np.zeros((m,)), 
                                   mu_w=np.zeros((n,)), use_Taylor=use_Taylor)
            decimation_factor = decimation_factor_rossler

        print("Test data generated using sigma_e2: {} dB, SMNR: {} dB".format(sigma_e2_dB_test, smnr_dB_test))
        
        idx_test = 0
        while idx_test <  N_test:
            x_ssm_i, y_ssm_i, cw_ssm_i = ssm_model_test.generate_single_sequence(T=int(T_test*decimation_factor), sigma_e2_dB=sigma_e2_dB_test, smnr_dB=smnr_dB_test)
            if np.isnan(x_ssm_i).any() == False:
                X[idx_test, :, :] = torch.from_numpy(x_ssm_i).type(torch.FloatTensor)
                Y[idx_test, :, :] = torch.from_numpy(y_ssm_i).type(torch.FloatTensor)
                Cw[idx_test, :, :] = torch.from_numpy(cw_ssm_i).type(torch.FloatTensor)
                idx_test += 1

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["Cw"] = Cw
        test_data_dict["model"] = ssm_model_test
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, ssm_type_test, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl", test_data_file.split('/')[-1])
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        Cw = test_data_dict["Cw"]
        ssm_model_test = test_data_dict["model"]
        f_fn, f_ukf_fn = get_f_function(ssm_type)

    print("*"*100)
    print("*"*100, file=orig_stdout)
    #i_test = np.random.choice(N_test)
    print("sigma_e2: {}dB, smnr: {}dB".format(sigma_e2_dB_test, smnr_dB_test))
    print("sigma_e2: {}dB, smnr: {}dB".format(sigma_e2_dB_test, smnr_dB_test), file=orig_stdout)

    ###########################################################################################################
    # NOTE: The following couple lines of code are only for rapid testing for debugging of code,
    # We take the first two samples from the testing dataset and pass them on to the prediction techniques
    idx = np.random.choice(X.shape[0],2, replace=False)
    print(idx, file=orig_stdout)
    Y = Y[idx]
    X = X[idx]
    Cw = Cw[idx]
    ###########################################################################################################

    # Collecting all estimator results
    X_estimated_dict = dict.fromkeys(models_list, {})

    #####################################################################################################################################################################
    # Estimator baseline: Least Squares
    N_test, Ty, dy = Y.shape
    N_test, Tx, dx = X.shape

    if 'ls' in models_list:
        # Get the estimate using the least-squares (LS) baseline!
        H_tensor = torch.from_numpy(jacobian(ssm_model_test.h_fn, torch.randn(ssm_model_test.n_states,)).numpy()).type(torch.FloatTensor)
        H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
        #X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)
        start_time_ls = timer()
        X_LS = torch.zeros_like(X)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                X_LS[i,j,:] = (torch.pinverse(H_tensor[i]) @ Y[i,j,:].reshape((dy, 1))).reshape((dx,))
        time_elapsed_ls = timer() - start_time_ls
        X_estimated_dict["ls"]["est"] = X_LS

    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Model-based filters 
    print("Fed to Model-based filters: ", file=orig_stdout)
    print("sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(sigma_e2_dB_test, smnr_dB_test, delta_t), file=orig_stdout)

    print("Fed to Model-based filters: ")
    print("sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(sigma_e2_dB_test, smnr_dB_test, delta_t))

    ssm_model_test.sigma_e2 = dB_to_lin(sigma_e2_dB_test)
    ssm_model_test.setStateCov(sigma_e2=dB_to_lin(sigma_e2_dB_test))

    # Estimator: EKF
    if "ekf" in models_list:
        print("Testing EKF ...", file=orig_stdout)

        # print(X, file=orig_stdout)
        # print(Y, file=orig_stdout)
        # print(ssm_model_test, file=orig_stdout)
        # print(f_fn, file=orig_stdout)  # this is a function
        # print(Cw, file=orig_stdout)
        # print(device, file=orig_stdout)  # cpu
        # print(use_Taylor, file=orig_stdout)  # False

        X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf, time_elapsed_ekf = test_ekf_ssm(
            X_test=X, Y_test=Y, ssm_model_test=ssm_model_test, f_fn=f_fn, Cw_test=Cw, 
            device=device, use_Taylor=use_Taylor
        )



        X_estimated_dict["ekf"]["est"] = X_estimated_ekf
        X_estimated_dict["ekf"]["est_cov"] = Pk_estimated_ekf
    else:
        X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf, time_elapsed_ekf = None, None, None, None

    # Estimator: UKF
    if "ukf" in models_list:
        print("Testing UKF ...", file=orig_stdout)
        X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf, time_elapsed_ukf = test_ukf_ssm(
            X_test=X, Y_test=Y, ssm_model_test=ssm_model_test, f_ukf_fn=f_ukf_fn, 
            Cw_test=Cw, device=device
        )
        X_estimated_dict["ukf"]["est"] = X_estimated_ukf
        X_estimated_dict["ukf"]["est_cov"] = Pk_estimated_ukf

        X_estimated_ukf_pred, Pk_estimated_ukf_pred, _, _, _, _, _, _, _ = test_ukf_ssm_one_step(
            X_test=X, Y_test=Y, ssm_model_test=ssm_model_test, f_ukf_fn=f_ukf_fn, 
            Cw_test=Cw, device=device
        )  

        Y_estimated_ukf_pred, Py_estimated_ukf_pred = get_y_pred_linear_meas(X_estimated_pred_test=X_estimated_ukf_pred,
                                                                            Pk_estimated_pred_test=Pk_estimated_ukf_pred,
                                                                            Cw_test=Cw, 
                                                                            ssm_model_test=ssm_model_test
                                                                        )

    else:
        X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf, time_elapsed_ukf = None, None, None, None
        Y_estimated_ukf_pred, Py_estimated_ukf_pred = None, None
    #####################################################################################################################################################################
    
    #####################################################################################################################################################################
    # Estimator: DANSE
    if "danse" in models_list:
        print("Testing DANSE ...", file=orig_stdout)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered, time_elapsed_danse = test_danse_ssm(
            Y_test=Y, ssm_model_test=ssm_model_test, model_file_saved_danse=model_file_saved_danse,
            Cw_test=Cw, rnn_type=rnn_type, device=device
        )
        X_estimated_dict["danse"]["est"] = X_estimated_filtered
        X_estimated_dict["danse"]["est_cov"] = Pk_estimated_filtered
        Y_estimated_pred, Py_estimated_pred = get_y_pred_linear_meas(X_estimated_pred_test=X_estimated_pred,
                                                                            Pk_estimated_pred_test=Pk_estimated_pred,
                                                                            Cw_test=Cw,
                                                                            ssm_model_test=ssm_model_test
                                                                        )

    else:
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered, time_elapsed_danse = None, None, None, None, None
        Y_estimated_pred, Py_estimated_pred = None, None
    
    #####################################################################################################################################################################

    # Estimator: IDANSE, need to modify the test_idanse_ssm function
    if "idanse" in models_list:
        print("Testing IDANSE ...", file=orig_stdout)
        # print(model_file_saved_idanse_list, file=orig_stdout)
        X_estimated_pred_idanse, Pk_estimated_pred_idanse, X_estimated_filtered_idanse, Pk_estimated_filtered_idanse, time_elapsed_idanse = test_idanse_ssm(
            Y_test=Y, ssm_model_test=ssm_model_test, model_file_saved_idanse_list=model_file_saved_idanse_list,
            Cw_test=Cw, rnn_type=rnn_type, device=device
        )
        X_estimated_dict["idanse"]["est"] = X_estimated_filtered_idanse
        X_estimated_dict["idanse"]["est_cov"] = Pk_estimated_filtered_idanse
        Y_estimated_pred_idanse, Py_estimated_pred_idanse = get_y_pred_linear_meas(X_estimated_pred_test=X_estimated_pred_idanse,
                                                                     Pk_estimated_pred_test=Pk_estimated_pred,
                                                                     Cw_test=Cw,
                                                                     ssm_model_test=ssm_model_test
                                                                     )

    else:
        X_estimated_pred_idanse, Pk_estimated_pred_idanse, X_estimated_filtered_idanse, Pk_estimated_filtered_idanse, time_elapsed_idanse_idanse = None, None, None, None, None
        Y_estimated_pred_idanse, Py_estimated_pred_idanse = None, None

    #####################################################################################################################################################################
    # Estimator: DANSE Supervised
    if "danse_supervised" in models_list:
        print("Testing DANSE (Supervised) ...", file=orig_stdout)
        X_estimated_pred_danse_supervised, Pk_estimated_pred_danse_supervised, X_estimated_filtered_danse_supervised, \
            Pk_estimated_filtered_danse_supervised, time_elapsed_danse_supervised = test_danse_supervised_ssm(
            Y_test=Y, ssm_model_test=ssm_model_test, model_file_saved_danse_supervised=model_file_saved_danse_supervised, 
            Cw_test=Cw, rnn_type=rnn_type, device=device
        )
        X_estimated_dict["danse_supervised"]["est"] = X_estimated_filtered_danse_supervised
        X_estimated_dict["danse_supervised"]["est_cov"] = Pk_estimated_filtered_danse_supervised
    else:
        X_estimated_pred_danse_supervised, Pk_estimated_pred_danse_supervised, X_estimated_filtered_danse_supervised, \
            Pk_estimated_filtered_danse_supervised, time_elapsed_danse_supervised = None, None, None, None, None
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Estimator: DANSE Semisupervised
    if "danse_semisupervised" in models_list:
        print("Testing DANSE (Semi-Supervised) ...", file=orig_stdout)
        X_estimated_pred_danse_semisupervised, Pk_estimated_pred_danse_semisupervised, X_estimated_filtered_danse_semisupervised,\
            Pk_estimated_filtered_danse_semisupervised, time_elapsed_danse_danse_semisupervised = test_danse_semisupervised_ssm(
            Y_test=Y, ssm_model_test=ssm_model_test, model_file_saved_danse_semisupervised=model_file_saved_danse_semisupervised, 
            Cw_test=Cw, rnn_type=rnn_type, device=device
        )
        X_estimated_dict["danse_semisupervised"]["est"] = X_estimated_filtered_danse_semisupervised
        X_estimated_dict["danse_semisupervised"]["est_cov"] = Pk_estimated_filtered_danse_semisupervised
        Y_estimated_pred_danse_semisupervised, Py_estimated_pred_danse_semisupervised = get_y_pred_linear_meas(X_estimated_pred_test=X_estimated_pred_danse_semisupervised,
                                                                            Pk_estimated_pred_test=Pk_estimated_pred_danse_semisupervised,
                                                                            Cw_test=Cw, 
                                                                            ssm_model_test=ssm_model_test
                                                                        )
    else:
       X_estimated_pred_danse_semisupervised, Pk_estimated_pred_danse_semisupervised, X_estimated_filtered_danse_semisupervised,\
            Pk_estimated_filtered_danse_semisupervised, time_elapsed_danse_danse_semisupervised = None, None, None, None, None
       Y_estimated_pred_danse_semisupervised, Py_estimated_pred_danse_semisupervised = None, None 
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    ####################################################################################################################################################

    #####################################################################################################################################################################
    # Estimator: DMM - ST-L
    ###################################################################################################################################################

    #####################################################################################################################################################################
    # Metrics calculation
    metrics_dict_for_one_smnr = dict.fromkeys(metrics_list, {})

    for metric_name in metrics_list:
        metric_fn = metric_to_func_map(metric_name=metric_name) if metric_name != "time_elapsed" else None

        if not metric_fn is None:
            metrics_dict_for_one_smnr[metric_name] = {
                "ls":metric_fn(X[:,:,:], X_LS[:,:,:]).numpy().item() if "ls" in models_list else None,
                "ekf": metric_fn(X[:,:,:], X_estimated_ekf[:,:,:]).numpy().item() if "ekf" in models_list else None,
                "ukf": metric_fn(X[:,:,:], X_estimated_ukf[:,:,:]).numpy().item() if "ukf" in models_list else None,
                "danse": metric_fn(X[:,:,:], X_estimated_filtered[:,0:,:]).numpy().item() if "danse" in models_list else None,
                "idanse": metric_fn(X[:, :, :], X_estimated_filtered_idanse[:, 0:, :]).numpy().item() if "idanse" in models_list else None,
            }

    metrics_dict_for_one_smnr["time_elapsed"] = {
                "ls":time_elapsed_ls if "ls" in models_list else None,
                "ekf": time_elapsed_ekf if "ekf" in models_list else None,
                "ukf": time_elapsed_ukf if "ukf" in models_list else None,
                "danse": time_elapsed_danse if "danse" in models_list else None,
                "idanse": time_elapsed_idanse if "idanse" in models_list else None,
            }
    
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Displaying metrics (logging and on-console print)
    for model_name in models_list:
        
        # Logs metrics
        print("{}, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(
            model_name, N_test, metrics_dict_for_one_smnr["nmse"][model_name], metrics_dict_for_one_smnr["nmse_std"][model_name],
            metrics_dict_for_one_smnr["mse_dB"][model_name], metrics_dict_for_one_smnr["mse_dB_std"][model_name], 
            metrics_dict_for_one_smnr["time_elapsed"][model_name]
            ))

        # On-console printing of metrics
        print("{}, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(
            model_name, N_test, metrics_dict_for_one_smnr["nmse"][model_name], metrics_dict_for_one_smnr["nmse_std"][model_name],
            metrics_dict_for_one_smnr["mse_dB"][model_name], metrics_dict_for_one_smnr["mse_dB_std"][model_name], 
            metrics_dict_for_one_smnr["time_elapsed"][model_name]
            ),
        file=orig_stdout)
    #####################################################################################################################################################################
    
    #####################################################################################################################################################################
    # Plot the result
    plot_3d_state_trajectory(
        X=torch.squeeze(X[0, :, :], 0).numpy(), 
        legend='$\\mathbf{x}^{true}$', 
        m='b-', 
        savefig_name="./figs/{}/{}/{}_x_true_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test), savefig=True)
     
    plot_3d_measurement_trajectory(
        Y=torch.squeeze(Y[0, :, :], 0).numpy(), 
        legend='$\\mathbf{y}^{true}$', 
        m='r-', 
        savefig_name="./figs/{}/{}/{}_y_true_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test), savefig=True)
    
    if "danse" in models_list:
        plot_3d_state_trajectory(
            X=torch.squeeze(X_estimated_filtered[0], 0).numpy(), 
            legend='$\\hat{\mathbf{x}}_{DANSE}$', 
            m='k-', 
            savefig_name="./figs/{}/{}/{}_x_danse_sigmae2_{}dB_smnr_{}dB.pdf".format(
                dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test), savefig=True)

    if "idanse" in models_list:
        plot_3d_state_trajectory(
            X=torch.squeeze(X_estimated_filtered_idanse[0], 0).numpy(),
            legend='$\\hat{\mathbf{x}}_{IDANSE}$',
            m='k-',
            savefig_name="./figs/{}/{}/{}_x_idanse_sigmae2_{}dB_smnr_{}dB.pdf".format(
                dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test), savefig=True)
    
    if "ukf" in models_list:
        plot_3d_state_trajectory(
            X=torch.squeeze(X_estimated_ukf[0], 0).numpy(), 
            legend='$\\hat{\mathbf{x}}_{UKF}$', 
            m='k-', 
            savefig_name="./figs/{}/{}/{}_x_ukf_sigmae2_{}dB_smnr_{}dB.pdf".format(
                dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test), savefig=True)
    

    plot_state_trajectory_axes_all(
        X=torch.squeeze(X[0,:,:],0).numpy(), 
        X_est_EKF=torch.squeeze(X_estimated_ekf[0,:,:],0).numpy() if "ekf" in models_list else None, 
        X_est_UKF=torch.squeeze(X_estimated_ukf[0,:,:],0).numpy() if "ukf" in models_list else None, 
        X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0).numpy() if "danse" in models_list else None,
        X_est_IDANSE=torch.squeeze(X_estimated_filtered_idanse[0], 0).numpy() if "idanse" in models_list else None,
        savefig=True,
        savefig_name="./figs/{}/{}/AxesAll_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test)
    )
    
    plot_state_trajectory_w_lims(
        X=torch.squeeze(X[0,:,:],0).numpy(), 
        X_est_UKF=torch.squeeze(X_estimated_ukf[0,:,:],0).numpy() if "ukf" in models_list else None, 
        X_est_UKF_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_ukf[0,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()) if "ukf" in models_list else None, 
        X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0).numpy() if "danse" in models_list else None,
        X_est_DANSE_std=np.sqrt(torch.diagonal(torch.squeeze(Pk_estimated_filtered[0,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()) if "danse" in models_list else None,
        savefig=True,
        savefig_name="./figs/{}/{}/Axes_w_lims_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test)
    )

    plot_meas_trajectory_w_lims(
        Y=torch.squeeze(Y[0,:,:],0).numpy(), 
        Y_pred_UKF=torch.squeeze(Y_estimated_ukf_pred[0,:,:], 0).numpy(), 
        Y_pred_UKF_std=np.sqrt(torch.diagonal(torch.squeeze(Py_estimated_ukf_pred[0,:,:,:], 0), offset=0, dim1=1,dim2=2).numpy()), 
        Y_pred_DANSE=torch.squeeze(Y_estimated_pred[0], 0).numpy(), 
        Y_pred_DANSE_std=np.sqrt(torch.diagonal(torch.squeeze(Py_estimated_pred[0], 0), offset=0, dim1=1,dim2=2).numpy()), 
        savefig=True,
        savefig_name="./figs/{}/{}/Meas_y_lims_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test)
    )
    
    #plt.show()
    sys.stdout = orig_stdout
    return metrics_dict_for_one_smnr

if __name__ == "__main__":

    # Testing parameters
    ssm_name = "Lorenz"
    m = 3
    n = 3
    T_train = 100
    N_train = 1000
    T_test = 2000
    N_test = 100
    sigma_e2_dB_test = -10.0
    device = 'cpu'
    nsup = 20
    bias = None  # By default should be positive, equal to 10.0
    p = None  # Keep this fixed at zero for now, equal to 0.0
    mode = 'full'
    if mode == 'low' or mode == 'high':
        evaluation_mode = 'partial_opt_{}_bias_{}_p_{}'.format(mode, bias, p)
    else:
        bias = None
        p = None
        evaluation_mode = 'ModTest_diff_smnr_nsup_{}_Ntrain_{}_Ttrain_{}_refactored'.format(nsup, N_train, T_train)

    ssmtype = "{}SSMn{}".format(ssm_name, n) if n < m else "{}SSM".format(ssm_name) # Hardcoded for {}SSMrn{} (random H low rank), {}SSMn{} (deterministic H low rank), 
    dirname = "{}SSMn{}x{}".format(ssm_name,m,n)
    os.makedirs('./figs/{}/{}'.format(dirname, evaluation_mode), exist_ok=True)

    smnr_dB_arr = np.array([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]) # , 0.0, 10.0, 20.0, 30.0
    smnr_dB_dict_arr = ["{}dB".format(smnr_dB) for smnr_dB in smnr_dB_arr]

    list_of_models_comparison = ["ekf", "ukf", "danse", "idanse"] #  "danse_semisupervised"] "ekf", "ukf", "danse"
    list_of_display_fmts = ["gp-.", "rd--", "ks--", "bo-", "mo-", "ys-", "co-"]
    list_of_metrics = ["nmse", "nmse_std", "mse_dB", "mse_dB_std", "time_elapsed"]

    #metrics_dict = dict.fromkeys(list_of_metrics, {})

    #for metric in list_of_metrics:
    #    for model_name in list_of_models_comparison:
    #        metrics_dict[metric][model_name] = np.zeros((len(smnr_dB_arr)))
    
    metrics_multidim_mat = np.zeros((len(list_of_metrics), len(list_of_models_comparison), len(smnr_dB_arr)))

    model_file_saved_dict = {
                            "danse":dict.fromkeys(smnr_dB_dict_arr, {}),
                            "idanse":dict.fromkeys(smnr_dB_dict_arr, {})
                            #"kalmannet":dict.fromkeys(smnr_dB_dict_arr, {}),
                            #"dmm_st-l":dict.fromkeys(smnr_dB_dict_arr, {}),
                            #"danse_supervised":dict.fromkeys(smnr_dB_dict_arr, {}), 
                            # "danse_semisupervised":dict.fromkeys(smnr_dB_dict_arr, {}),
                            }
    
    test_data_file_dict = {}

    for j, smnr_dB_label in enumerate(smnr_dB_dict_arr):
        test_data_file_dict[smnr_dB_label] = "../data/synthetic_data/test_trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(
            m, n, ssmtype, T_test, N_test, sigma_e2_dB_test, smnr_dB_arr[j]
            )
    
    for key in model_file_saved_dict.keys():
        for j, smnr_dB_label in enumerate(smnr_dB_dict_arr):

            if key == "kalmannet":
                saved_model = "KNetUoffline"
            else:
                saved_model = key
            
            N_train_actual = nsup if not nsup is None and saved_model in ["danse_supervised"] else N_train 

            if saved_model == "danse_semisupervised" and not nsup is None:
                model_file_saved_dict[key][smnr_dB_label] = glob.glob("../run/models/*{}_{}_*nsup_{}_m_{}_n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                    ssmtype, saved_model, nsup, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))[-1]
                # pass
            elif saved_model == "KNetUoffline":
                model_file_saved_dict[key][smnr_dB_label] = glob.glob("../run/models/*{}_{}_*m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                    ssmtype, saved_model, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))[0]
            elif saved_model == "danse":
                # print("../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                #     ssmtype, saved_model, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))
                # print(glob.glob("../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                #     ssmtype, saved_model, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))[-1])
                # print(glob.glob(
                #     "../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB/*idanse_gru*".format(
                #         ssmtype, saved_model, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j])))
                model_file_saved_dict[key][smnr_dB_label] = glob.glob("../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*danse*".format(
                    ssmtype, saved_model, m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))[0]


            elif saved_model == "idanse":
                model_file_saved_dict[key][smnr_dB_label] = \
                glob.glob("../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/idanse_gru*".format(
                    ssmtype, 'danse', m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j]))
                # print(glob.glob("../run/models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*idanse_gru*".format(
                #     ssmtype, 'danse', m, n, T_train, N_train_actual, sigma_e2_dB_test, smnr_dB_arr[j])))

    test_logfile = "../log/{}_test_{}_T_{}_N_{}_ModTest.log".format(ssmtype, evaluation_mode, T_test, N_test)
    test_jsonfile = "../log/{}_test_{}_T_{}_N_{}_ModTest.json".format(ssmtype, evaluation_mode, T_test, N_test)

    for idx_smnr, smnr_dB_label in enumerate(smnr_dB_dict_arr):
        
        test_data_file_i = test_data_file_dict[smnr_dB_label]
        learnable_model_files_i = {key: model_file_saved_dict[key][smnr_dB_label] for key in model_file_saved_dict.keys()}

        metrics_dict_i = test_on_ssm_model(device=device, learnable_model_files = learnable_model_files_i, test_data_file=test_data_file_i, 
            test_logfile=test_logfile, evaluation_mode=evaluation_mode, metrics_list=list_of_metrics, models_list=list_of_models_comparison,
            dirname=dirname)
        
        for idx_metric, idx_model_name in itertools.product(list(range(len(list_of_metrics))), list(range(len(list_of_models_comparison)))):
            metrics_multidim_mat[idx_metric, idx_model_name, idx_smnr] = metrics_dict_i[list_of_metrics[idx_metric]][list_of_models_comparison[idx_model_name]]

    metrics_dict = {}
    metrics_dict['result_mat'] = metrics_multidim_mat
    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(metrics_dict, cls=NDArrayEncoder, indent=2))  
    
    plt.rcParams['font.family'] = 'serif'
    display_metrics = [idx_metric for idx_metric in range(len(list_of_metrics)) if not "_std" in list_of_metrics[idx_metric]]
    

    for idx_display_metric in range(len(display_metrics)):

        if list_of_metrics[idx_display_metric] != "time_elapsed":
            list_of_models_comparison = ["danse", "idanse"]  # "ekf", "ukf",
            plt.figure()
            for j, model_name in enumerate(list_of_models_comparison):

                plt.errorbar(smnr_dB_arr, metrics_multidim_mat[idx_display_metric, j+2, :], fmt=list_of_display_fmts[j+2], yerr=metrics_multidim_mat[idx_display_metric+1, j+2, :],  linewidth=1.5, label="{}".format(model_name.upper()))

            plt.xlabel('SMNR (in dB)')
            plt.ylabel('{} (in dB)'.format(list_of_metrics[idx_display_metric].upper()))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            #plt.subplot(212)
            tikzplotlib.save('./figs/{}/{}/{}_vs_SMNR_{}_idanse_vs_danse.tex'.format(dirname, evaluation_mode, list_of_metrics[idx_display_metric].upper(), ssm_name))
            plt.savefig('./figs/{}/{}/{}_vs_SMNR_{}_idanse_vs_danse.pdf'.format(dirname, evaluation_mode, list_of_metrics[idx_display_metric].upper(), ssm_name))

            plt.figure()
            list_of_models_comparison = ["ekf", "ukf", "danse", "idanse"]  # "ekf", "ukf",
            for j in [0, 1, 3]:
                plt.errorbar(smnr_dB_arr, metrics_multidim_mat[idx_display_metric, j, :],
                             fmt=list_of_display_fmts[j],
                             yerr=metrics_multidim_mat[idx_display_metric + 1, j, :], linewidth=1.5,
                             label="{}".format(list_of_models_comparison[j].upper()))

            plt.xlabel('SMNR (in dB)')
            plt.ylabel('{} (in dB)'.format(list_of_metrics[idx_display_metric].upper()))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.subplot(212)
            tikzplotlib.save('./figs/{}/{}/{}_vs_SMNR_{}_idanse_vs_kfs.tex'.format(dirname, evaluation_mode,
                                                                     list_of_metrics[idx_display_metric].upper(),
                                                                     ssm_name))
            plt.savefig('./figs/{}/{}/{}_vs_SMNR_{}_idanse_vs_kfs.pdf'.format(dirname, evaluation_mode,
                                                                list_of_metrics[idx_display_metric].upper(), ssm_name))

        else:
            plt.figure()
            for j, model_name in enumerate(list_of_models_comparison):
                plt.plot(smnr_dB_arr, metrics_multidim_mat[idx_display_metric, j, :], list_of_display_fmts[j], linewidth=1.5, label="{}".format(model_name.upper()))
            plt.xlabel('SMNR (in dB)')
            plt.ylabel('{} (in secs)'.format(list_of_metrics[idx_display_metric].upper()))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            #plt.subplot(212)
            tikzplotlib.save('./figs/{}/{}/{}_vs_SMNR_{}.tex'.format(dirname, evaluation_mode, list_of_metrics[idx_display_metric].upper(), ssm_name))
            plt.savefig('./figs/{}/{}/{}_vs_SMNR_{}.pdf'.format(dirname, evaluation_mode, list_of_metrics[idx_display_metric].upper(), ssm_name))
