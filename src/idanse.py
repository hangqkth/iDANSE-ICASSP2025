#####################################################
# Creators: Hang Qin
# Sep 2024
#####################################################
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim, distributions
from timeit import default_timer as timer
import sys
import copy
import math
import os
from utils.utils import compute_log_prob_normal, create_diag, compute_inverse, count_params, ConvergenceMonitor
# from utils.plot_functions import plot_state_trajectory, plot_state_trajectory_axes
import torch.nn.functional as F
from src.rnn_idanse import RNN_model


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    return None


def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets


class IDANSE(nn.Module):

    def __init__(self, n_states, n_obs, mu_w, C_w, H, mu_x0, C_x0, batch_size, rnn_type, rnn_params_dict, device='cpu'):
        super(IDANSE, self).__init__()

        self.device = device

        # Initialize the paramters of the state estimator
        self.n_states = n_states
        self.n_obs = n_obs

        # Initializing the parameters of the initial state
        self.mu_x0 = self.push_to_device(mu_x0)
        self.C_x0 = self.push_to_device(C_x0)

        # Initializing the parameters of the measurement noise
        self.mu_w = self.push_to_device(mu_w)
        self.C_w = self.push_to_device(C_w)

        # Initialize the observation model matrix
        self.H = self.push_to_device(H)
        # self.H = torch.nn.Parameter(torch.randn(n_obs, n_states), requires_grad=True).to(self.device)

        self.batch_size = batch_size

        # Initialize RNN type
        self.rnn_type = rnn_type

        # Initialize the parameters of the RNN
        self.rnn = RNN_model(**rnn_params_dict[self.rnn_type]).to(self.device)

        # Initialize various means and variances of the estimator

        # Prior parameters
        self.mu_xt_yt_current = None
        self.L_xt_yt_current = None

        # Marginal parameters
        self.mu_yt_current = None
        self.L_yt_current = None

        # Posterior parameters
        self.mu_xt_yt_prev = None
        self.L_xt_yt_prev = None


    def push_to_device(self, x):
        """ Push the given tensor to the device
        """
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def compute_prior_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        self.mu_xt_yt_prev = mu_xt_yt_prev
        self.L_xt_yt_prev = create_diag(L_xt_yt_prev)
        return self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev, Cw_batch):
        Cw_batch_seq = torch.repeat_interleave(Cw_batch.unsqueeze(1), L_xt_yt_prev.shape[1], dim=1)
        # print(self.H.device, self.mu_xt_yt_prev.device, self.mu_w.device)
        self.mu_yt_current = torch.einsum('ij,ntj->nti', self.H, mu_xt_yt_prev) + self.mu_w.squeeze(-1)
        self.L_yt_current = self.H @ L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + Cw_batch_seq  # + self.C_w

    def compute_posterior_mean_vars(self, Yi_batch, Cw_batch):
        Cw_batch_seq = torch.repeat_interleave(Cw_batch.unsqueeze(1), self.L_xt_yt_prev.shape[1], dim=1)
        # print((self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1)).shape, self.L_xt_yt_prev.shape, Cw_batch_seq.shape)
        Re_t_inv = torch.inverse(self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0,
                                                                              1) + Cw_batch_seq)  # torch.inverse(self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w)
        self.K_t = (self.L_xt_yt_prev @ (self.H.T @ Re_t_inv))
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.einsum('ntij,ntj->nti', self.K_t, (
                    Yi_batch - torch.einsum('ij,ntj->nti', self.H, self.mu_xt_yt_prev)))
        # self.L_xt_yt_current = self.L_xt_yt_prev - (torch.einsum('ntij,ntjk->ntik',
        #                    self.K_t, self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w) @ torch.transpose(self.K_t, 2, 3))
        self.L_xt_yt_current = self.L_xt_yt_prev - (torch.einsum('ntij,ntjk->ntik',
                                                                 self.K_t,
                                                                 self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0,
                                                                                                              1) + Cw_batch_seq) @ torch.transpose(
            self.K_t, 2, 3))

        return self.mu_xt_yt_current, self.L_xt_yt_current

    def compute_logpdf_Gaussian(self, Y):
        _, T, _ = Y.shape
        logprob = - 0.5 * self.n_obs * T * math.log(math.pi * 2) - 0.5 * torch.logdet(self.L_yt_current).sum(1) \
                  - 0.5 * torch.einsum('nti,nti->nt',
                                       (Y - self.mu_yt_current),
                                       torch.einsum('ntij,ntj->nti', torch.inverse(self.L_yt_current),
                                                    (Y - self.mu_yt_current))).sum(1)

        return logprob

    def compute_predictions(self, Y_test_batch, X_hat_batch, Cw_test_batch):

        mu_x_given_Y_test_batch, vars_x_given_Y_test_batch = self.rnn.forward(y=Y_test_batch, x_hat=X_hat_batch)
        mu_xt_yt_prev_test, L_xt_yt_prev_test = self.compute_prior_mean_vars(
            mu_xt_yt_prev=mu_x_given_Y_test_batch,
            L_xt_yt_prev=vars_x_given_Y_test_batch
            )
        mu_xt_yt_current_test, L_xt_yt_current_test = self.compute_posterior_mean_vars(Yi_batch=Y_test_batch, Cw_batch=Cw_test_batch)
        return mu_xt_yt_prev_test, L_xt_yt_prev_test, mu_xt_yt_current_test, L_xt_yt_current_test

    def forward(self, Yi_batch, X_hat_batch, Cw_batch):

        mu_batch, vars_batch = self.rnn.forward(y=Yi_batch, x_hat=X_hat_batch)
        mu_xt_yt_prev, L_xt_yt_prev = self.compute_prior_mean_vars(mu_xt_yt_prev=mu_batch, L_xt_yt_prev=vars_batch)
        self.compute_marginal_mean_vars(mu_xt_yt_prev=mu_xt_yt_prev, L_xt_yt_prev=L_xt_yt_prev, Cw_batch=Cw_batch)
        logprob_batch = self.compute_logpdf_Gaussian(Y=Yi_batch) / (Yi_batch.shape[1] * Yi_batch.shape[2])  # Per dim. and per sequence length
        log_pYT_batch_avg = logprob_batch.mean(0)

        return log_pYT_batch_avg


def train(train_loader, model, optimizer, x_hat_list, k, epoch, device, tr_running_loss, tr_loss_epoch_sum):
    for i, data in enumerate(train_loader, 0):

        model.train()
        tr_Y_batch, tr_X_batch, tr_Cw_batch = data
        optimizer.zero_grad()
        Y_train_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)  # why requires_grad=False in training?
        Cw_train_batch = Variable(tr_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)

        if k == 0:
            x_hat_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            x_hat_list.append(x_hat_batch)
        else:
            x_hat_batch = x_hat_list[i]


        log_pY_train_batch = -model.forward(Y_train_batch, x_hat_batch, Cw_train_batch)
        log_pY_train_batch.backward()

        optimizer.step()

        # print statistics
        tr_running_loss += log_pY_train_batch.item()
        tr_loss_epoch_sum += log_pY_train_batch.item()

        if i % 100 == 99 and ((epoch + 1) % 100 == 0):  # print every 10 mini-batches
            # print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100))
            # print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100), file=orig_stdout)
            tr_running_loss = 0.0

    return tr_running_loss, tr_loss_epoch_sum, x_hat_list


def evaluation(val_loader, model, x_hat_list, k, device, mse_criterion, val_loss_epoch_sum, val_mse_loss_epoch_sum):
    # evaluation
    with torch.no_grad():

        for i, data in enumerate(val_loader, 0):
            val_Y_batch, val_X_batch, val_Cw_batch = data
            Y_val_batch = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            Cw_val_batch = Variable(val_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)

            if k == 0:
                x_hat_val = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
                x_hat_list.append(x_hat_val)
            else:
                x_hat_val = x_hat_list[i]

            val_mu_X_predictions_batch, val_var_X_predictions_batch, val_mu_X_filtered_batch, val_var_X_filtered_batch = model.compute_predictions(
                Y_val_batch, x_hat_val, Cw_val_batch)
            log_pY_val_batch = -model.forward(Y_val_batch, x_hat_val, Cw_val_batch)
            val_loss_epoch_sum += log_pY_val_batch.item()

            val_mse_loss_batch = mse_criterion(val_X_batch.to(device), val_mu_X_filtered_batch)
            # print statistics
            val_mse_loss_epoch_sum += val_mse_loss_batch.item()
    return val_loss_epoch_sum, val_mse_loss_epoch_sum, x_hat_list


def update_x_hat(model, train_loader, val_loader, x_hat_train_list, x_hat_val_list, device):
    """Update X_hat_batch"""
    model.eval()
    with torch.no_grad():
        # update X_hat_batch
        for i, data in enumerate(train_loader, 0):
            tr_Y_batch, tr_X_batch, tr_Cw_batch = data
            Y_train_batch = Variable(tr_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            Cw_train_batch = Variable(tr_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            _, _, mu_X_filtered_batch, _ = model.compute_predictions(Y_train_batch, x_hat_train_list[i], Cw_train_batch)
            x_hat_train_list[i] = mu_X_filtered_batch

        # update X_hat_val
        for i, data in enumerate(val_loader, 0):
            val_Y_batch, val_X_batch, val_Cw_batch = data
            Y_val_batch = Variable(val_Y_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            Cw_val_batch = Variable(val_Cw_batch, requires_grad=False).type(torch.FloatTensor).to(device)
            _, _, mu_X_filtered_batch, _ = model.compute_predictions(Y_val_batch, x_hat_val_list[i], Cw_val_batch)
            x_hat_val_list[i] = mu_X_filtered_batch
    return x_hat_train_list, x_hat_val_list


def train_idanse(model, options, train_loader, val_loader, logfile_path, modelfile_path, save_chkpoints,
                device='cpu', tr_verbose=False):
    # Push the model to device and count parameters
    nepochs = 2000
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)

    # Set the model to training
    model.train()
    mse_criterion = nn.MSELoss()  # just for evaluation
    optimizer = optim.Adam(model.parameters(), lr=model.rnn.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//3, gamma=0.9) # gamma was initially 0.9
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs // 6, gamma=0.9)  # gamma is now set to 0.8
    tr_losses = []
    val_losses = []
    val_mse_losses = []
    epochs_at_k = []

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    # if save_chkpoints == True:
    if save_chkpoints == "all" or save_chkpoints == "some":
        # No grid search
        if logfile_path is None:
            training_logfile = "./log/idanse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path

    elif save_chkpoints == None:
        # Grid search
        if logfile_path is None:
            training_logfile = "./log/gs_training_idanse_{}.log".format(model.rnn_type)
        else:
            training_logfile = logfile_path

    # Call back parameters

    patience = 0
    num_patience = 3
    min_delta = options['rnn_params_dict'][model.rnn_type]["min_delta"]
    # 1e-3 for simpler model, for complicated model we use 1e-2
    # min_tol = 1e-3 # for tougher model, we use 1e-2, easier models we use 1e-5

    check_patience = False

    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, 'a')
    sys.stdout = f_tmp

    # Convergence monitoring (checks the convergence but not ES of the val_loss)
    model_monitor = ConvergenceMonitor(tol=min_delta, max_epochs=num_patience)

    # This checkes the ES of the val loss, if the loss deteriorates for specified no. of
    # max_epochs, stop the training
    # model_monitor = ConvergenceMonitor_ES(tol=min_tol, max_epochs=num_patience)

    print("------------------------------ Training begins --------------------------------- \n")
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params), file=orig_stdout)
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # Start time
    starttime = timer()

    smnr = modelfile_path.split('smnr_')[-1].replace('dB', '')
    k_max = 8


    try:
        x_hat_train_list, x_hat_val_list = [], []
        val_loss_k = np.inf


        for k in range(k_max):
            # for epoch in range(nepochs):
            # Convergence monitoring (checks the convergence but not ES of the val_loss)
            model_monitor = ConvergenceMonitor(tol=min_delta, max_epochs=num_patience)

            # Ensure a model is saved for every k
            best_val_loss = np.inf
            tr_loss_for_best_val_loss = np.inf
            best_model_wts = None
            best_val_epoch = None

            # reset models parameter
            # for layer in model.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            epoch_counter = 0
            for epoch in range(nepochs):
                epoch_counter += 1
                tr_running_loss = 0.0
                tr_loss_epoch_sum = 0.0
                val_loss_epoch_sum = 0.0
                val_mse_loss_epoch_sum = 0.0

                # train
                tr_running_loss, tr_loss_epoch_sum, x_hat_train_list = train(
                    train_loader, model, optimizer, x_hat_train_list, k, epoch, device, tr_running_loss, tr_loss_epoch_sum)

                scheduler.step()
                endtime = timer()

                # Measure wallclock time
                time_elapsed = endtime - starttime

                # evaluation
                val_loss_epoch_sum, val_mse_loss_epoch_sum, x_hat_val_list = evaluation(
                    val_loader, model, x_hat_val_list, k, device, mse_criterion, val_loss_epoch_sum, val_mse_loss_epoch_sum)

                # Loss at the end of each epoch
                tr_loss = tr_loss_epoch_sum / len(train_loader)
                val_loss = val_loss_epoch_sum / len(val_loader)
                val_mse_loss = val_mse_loss_epoch_sum / len(val_loader)

                # Record the validation loss per epoch
                if (epoch + 1) > nepochs // 100:
                    model_monitor.record(val_loss)
                    # print(val_loss, file=orig_stdout)

                # Displaying loss at an interval of 200 epochs
                if tr_verbose == True and (((epoch + 1) % 10) == 0 and ((k + 1) % 5) == 0 or epoch == 0):
                    print("Epoch: {}/{}, k: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE:{:.9f}".format(epoch + 1,
                        nepochs, k+1, k_max, tr_loss, val_loss, val_mse_loss), file=orig_stdout)
                    # save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))
                    print("Epoch: {}/{}, k: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE: {:.9f}, Time_Elapsed:{:.4f} secs".format(
                            epoch + 1, nepochs, k+1, k_max, tr_loss, val_loss, val_mse_loss, time_elapsed))


                # Saving every value
                tr_losses.append(tr_loss)
                val_losses.append(val_loss)
                val_mse_losses.append(val_mse_loss)

                # Check monitor flag
                # print(model_monitor.monitor(epoch=epoch + 1), file=orig_stdout)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss  # Save best validation loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_mse_loss = val_mse_loss

                if model_monitor.monitor(epoch=epoch + 1) == True:
                    if tr_verbose == True:
                        print("Training convergence attained! Saving model at Epoch: {}".format(epoch + 1), file=orig_stdout)

                    # print("Training convergence attained at Epoch: {}!".format(epoch + 1))
                    # Save the best model as per validation loss at the end
                    tr_loss_for_best_val_loss = tr_loss  # Training loss corresponding to best validation loss
                    # best_val_loss = val_loss  # Save best validation loss
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    # print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
                    # save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
                    break


            epochs_at_k.append(epoch_counter)

            if not best_model_wts is None:
                model_filename = "idanse_{}_ckpt_k_{}_best.pt".format(model.rnn_type, k + 1)
                torch.save(best_model_wts, model_filepath + "/" + model_filename)
                model.load_state_dict(torch.load(model_filepath + "/" + "idanse_{}_ckpt_k_{}_best.pt".format(model.rnn_type, k+1)))
            else:
                model_filename = "idanse_{}_ckpt_k_{}_best.pt".format(model.rnn_type, k + 1)
                save_model(model, model_filepath + "/" + model_filename)  # no need to load model since last epoch is the best
                model.load_state_dict(torch.load(model_filepath + "/" + "idanse_{}_ckpt_k_{}_best.pt".format(model.rnn_type, k+1)))
            # Update X_hat_batch
            print('saving the model at k={}, val_loss={}, val_mse_loss={}'.format(k + 1, best_val_loss, best_val_mse_loss), file=orig_stdout)
            x_hat_train_list, x_hat_val_list = update_x_hat(model, train_loader, val_loader, x_hat_train_list, x_hat_val_list, device)

    except KeyboardInterrupt:

        if tr_verbose == True:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch + 1), file=orig_stdout)
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch + 1))
        else:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch + 1))

        if not save_chkpoints is None:
            model_filename = "idanse_{}_ckpt_epoch_{}_latest.pt".format(model.rnn_type, epoch + 1)
            torch.save(model, model_filepath + "/" + model_filename)

    print("------------------------------ Training ends --------------------------------- \n")
    # Restoring the original std out pointer
    sys.stdout = orig_stdout

    return tr_losses, val_losses, val_mse_losses, epochs_at_k, best_val_loss,  tr_loss_for_best_val_loss, model




