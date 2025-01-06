import matplotlib.pyplot as plt
import numpy as np
import os
import json

ssm = 'Lorenz'
danse_log = './log_danse_'+ssm
idanse_log = './log_idanse_'+ssm

danse_logs = [os.path.join(danse_log, dir_path, 'danse_gru_losses_eps2000.json') for dir_path in os.listdir(danse_log)]
idanse_logs = [os.path.join(idanse_log, dir_path, 'danse_gru_losses_eps200.json') for dir_path in os.listdir(idanse_log)]

def read_danse_log(training_log):
    with open(training_log, 'r') as file:
        records = json.load(file)
    val_losses_list, val_mse_list = records['tr_losses'], records['val_losses']
    return val_losses_list, val_mse_list

def read_idanse_log(training_log):
    with open(training_log, 'r') as file:
        records = json.load(file)
    val_losses_list, val_mse_list, k_epochs_list = records['tr_losses'], records['val_losses'], records['epochs_at_k']
    return val_losses_list, val_mse_list, k_epochs_list


def plot_results(danse_log, idanse_log, loss_type):
    val_losses_danse, val_mse_danse = read_danse_log(danse_log)
    val_losses_idanse, val_mse_idanse, k_epochs = read_idanse_log(idanse_log)
    epochs_danse = list(range(1, len(val_losses_danse)+1))
    epochs_idanse = list(range(1, sum(k_epochs)+1))

    smnr = danse_log.split('smnr_')[-1].split('dB')[0]

    if loss_type == 'mse':
        losses_danse, losses_idanse = val_mse_danse, val_mse_idanse
    elif loss_type == 'nll':
        losses_danse, losses_idanse = val_losses_danse, val_losses_idanse
    else:
        losses_danse, losses_idanse = [], []
        exit()

    k_epoch = 1
    k_epochs_list = [k_epoch]
    for i in range(len(k_epochs)-1):
        k_epoch += k_epochs[i]
        k_epochs_list.append(k_epoch)

    # plt.figure(figsize=(8, 6))
    plt.plot(epochs_danse, losses_danse)
    plt.plot(epochs_idanse, losses_idanse)
    plt.xlim([0, 150])

    # print(k_epochs_list)
    # print(len(val_losses_idanse))
    for k in range(len(k_epochs_list)):
        plt.scatter(k_epochs_list[k], losses_idanse[k_epochs_list[k]-1], color='r', marker='x')
        plt.text(k_epochs_list[k]+0.5, losses_idanse[k_epochs_list[k]-1]+0.001, str(k+1), fontsize=12)


    plt.legend(['Danse', 'iDanse', 'k'], fontsize=15)
    # for i in range(len(ks)):
    #     plt.text(epochs_idanse_x[i], mse_idanse[i], str(ks[i]), fontsize=9)
    plt.ylabel('Val. '+loss_type.upper(), fontsize=15)
    plt.xlabel('epochs required', fontsize=15)
    plt.xticks(fontsize=14)  # Change the font size of the x-axis tick labels
    plt.yticks(fontsize=14)

    plt.grid(True)
    plt.title('SMNR=' + smnr +' dB', fontsize=15)
    # plt.show()


# for d in range(len(danse_logs)):
#     plot_results(danse_logs[d], idanse_logs[d], 'nll')

plt.figure(figsize=(10, 5))
data_ids = [0, 3]
loss_type = 'nll'  # or mse
print(danse_logs)
for i in range(len(data_ids)):
    plt.subplot(1, len(data_ids), i+1)
    plot_results(danse_logs[data_ids[i]], idanse_logs[data_ids[i]], 'nll')
plt.tight_layout()
plt.savefig('./visualize_training_{}.pdf'.format(loss_type))
plt.show()
