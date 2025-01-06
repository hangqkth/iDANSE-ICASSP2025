# iDANSE: Iterative Data-driven Nonlinear State Estimation of Model-free Hidden Sequences

**Accepted in 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)**

This is the project repository of iDANSE, an iterative nonlinear state estimation method for model-free hidden sequences. 

iDANSE is an extension of DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup. DANSE is accepted in IEEE Transactions on Signal Processing (IEEE-TSP) (March 2024)
([pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10485649))([DANSE repo](https://github.com/anubhabghosh/danse_jrnl))

## Authors
Hang Qin (hangq@kth.se), Anubhab Ghosh (anubhabg@kth.se), Saikat Chatterjee (sach@kth.se)

## Dependencies 
It is recommended to build an environment either in [`pip`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [`conda`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the following packages (I used `conda` as personal preference):
- PyTorch (1.10.0)
- Python (>= 3.7.0) with standard packages as part of an Anaconda installation such as Numpy, Scipy, Matplotlib, etc. The settings for the code were:
    - Numpy (1.21.6)
    - Matplotlib (3.5.1)
    - Scipy (1.7.3)
    - Scikit-learn (1.0.2)

- Filterpy (1.4.5) (for implementation of Unscented Kalman Filter (UKF)): [https://filterpy.readthedocs.io/en/latest/](https://filterpy.readthedocs.io/en/latest/)
- Tikzplotlib (for figures) [https://github.com/nschloe/tikzplotlib](https://github.com/nschloe/tikzplotlib)

## Datasets used 

The experiments were carried out using synthetic data generated with non-linear Lorenz attractor SSM.
Details about these models and their underlying dynamics can be found in `./bin/ssm_models.py`. 

## Reference models (implemented in PyTorch + Numpy)

- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- DANSE ([DANSE repo](https://github.com/anubhabghosh/danse_jrnl))

## GPU Support

The training-based methods: DANSE, iDANSE, were run on a NVIDIA GeForce RTX 3070 Laptop GPU. 

## Code organization
This would be the required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- main_idanse_opt.py (main function for training 'iDANSE' model)
- main_danse_opt.py (main function for training 'DANSE' model)
- parameters_opt.py (hyper-parameters)

- data/ (contains stored datasets in .pkl files)
| - synthetic_data/ (contains datasets related to SSM models in .pkl files)

- src/ (contains model-related files)
| - idanse.py (for training iDANSE)
| - danse.py (for training the unsupervised version of DANSE)
| - kf.py (for running the Kalman filter (KF) at test-time for inference)
| - ekf.py (for running the extended Kalman filter (EKF) at test-time for inference)
| - ukf_aliter.py (for running the unscented Kalman filter (UKF) at test-time for inference)
| - ukf_aliter_one_step.py (for running the unscented Kalman filter (UKF) at test-time for inference related to one-step ahead of forecasting!)
| - rnn.py (class definition of the RNN model for DANSE)
| - rnn_idanse.py (class definition of the RNN model for iDANSE)

- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)

- utils/ (contains helping functions)

- tests/ (contains files and functions for evaluation at test time)
| - figs/ (contains resulting model figures)
| - test_model_with_idanse.py (for testing iDANSE and other reference models: DANSE, EKF, UKF)

- bin/ (contains data generation files)
| - ssm_models.py (contains the classes for state space models)
| - generate_data.py (contains code for generating training datasets)

- run/ (folder containing the scripts to run the `main` scripts or data-generation scripts at one go for either different smnr_dB / sigma_e2_dB / N)
| - run_training.py (for training data-driven model: DANSE and iDANSE)
| - run_generate_data.py (for generating data)
| - visualizing_training.py
````

## Training

1. Generate data by running run/run_generate_data.py
2. Edit the hyper-parameters for the iDANSE architecture in `./parameters_opt.py`.
3. Run the training for iDANSE by running run/run_training.py

For the `datafile` and `splits` arguments:
`N` denotes the number of sample trajectories, `T` denotes the length of each sample trajectory. 




## Evaluation
1. For reproducing the experiments, make sure data is generated and both iDANSE and DANSE are trained for all SMNRs.
2. Run ./test/test_model+with_idanse.py, all figures and results can be found in ./test/figs
