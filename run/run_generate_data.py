import subprocess

# Configuration
PYTHON = "python"
N = 1000
T = 100
n_states = 3
n_obs = 3
dataset_type = "LorenzSSM"
script_name = "../bin/generate_data.py"
output_path = "../data/synthetic_data/"
sigma_e2_dB = -10.0

# List of SMNR values
smnr_dB_values = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]

def run_generate_data(smnr_dB):
    command = [
        PYTHON, script_name,
        "--n_states", str(n_states),
        "--n_obs", str(n_obs),
        "--num_samples", str(N),
        "--sequence_length", str(T),
        "--sigma_e2_dB", str(sigma_e2_dB),
        "--smnr_dB", str(smnr_dB),
        "--dataset_type", dataset_type,
        "--output_path", output_path
    ]
    subprocess.run(command)

# Run the data generation for each SMNR value
for smnr_dB in smnr_dB_values:
    run_generate_data(smnr_dB)