import subprocess


### Choose running model (danse or idanse) ###
run_model = 'idanse'  # 'danse'

# Configuration
PYTHON = "python"
N = 1000
T = 100
n_states = 3
n_obs = 3
dataset_type = "LorenzSSM"
script_name = "../main_"+run_model+"_opt.py"  # main_danse_opt.py or main_idanse_opt.py
output_path = "../data/synthetic_data/"
sigma_e2_dB = -10.0
rnn_model_type = "gru"

# List of SMNR values
smnr_dB_values = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]  # -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0

def run_training(smnr_dB):
    datafile_path = f"{output_path}/trajectories_m_{n_states}_n_{n_obs}_{dataset_type}_data_T_{T}_N_{N}_sigmae2_{sigma_e2_dB}dB_smnr_{smnr_dB}dB.pkl"
    splits_path = f"{output_path}/splits_m_{n_states}_n_{n_obs}_{dataset_type}_data_T_{T}_N_{N}_sigmae2_{sigma_e2_dB}dB_smnr_{smnr_dB}dB.pkl"
    command = [
        PYTHON, script_name,
        "--mode", "train",
        "--bidirectional", "False", # or "False"
        "--rnn_model_type", rnn_model_type,
        "--dataset_type", dataset_type,
        "--datafile", datafile_path,
        "--splits", splits_path,
    ]
    print(smnr_dB)
    subprocess.run(command)

# Run the data generation for each SMNR value
for smnr_dB in smnr_dB_values:
    run_training(smnr_dB)