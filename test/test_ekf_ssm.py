#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################

import sys
import os
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from parameters_opt import J_test
from src.ekf import EKF

def test_ekf_ssm(X_test, Y_test, ssm_model_test, f_fn, Cw_test=None, device='cpu', use_Taylor=True):

    # Initializing the extended Kalman filter model in PyTorch
    ekf_model = EKF(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        J=J_test,
        f=f_fn, #ssm_model_test.A_fn, f_lorenz for KalmanNet paper, f_lorenz_danse for our work
        h=ssm_model_test.h_fn,
        delta=ssm_model_test.delta,
        Q=ssm_model_test.Ce, #For KalmanNet
        R=ssm_model_test.Cw, # For KalmanNet
        device=device,
        use_Taylor=use_Taylor
    )

    # Get the estimates using an extended Kalman filter model
    
    X_estimated_ekf = None
    Pk_estimated_ekf = None

    start_time_ekf = timer()
    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = ekf_model.run_mb_filter(X_test, Y_test)
    time_elapsed_ekf = timer() - start_time_ekf

    return X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf, time_elapsed_ekf