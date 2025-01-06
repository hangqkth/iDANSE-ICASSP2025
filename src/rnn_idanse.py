import numpy as np
import scipy
import sys
import torch
from torch import nn
import torch.nn.functional as F


# Create an RNN model for prediction
class RNN_model(nn.Module):
    """ This super class defines the specific model to be used i.e. LSTM or GRU or RNN
    """

    def __init__(self, input_size, output_size, n_hidden, n_layers,
                 model_type, lr, num_epochs, n_hidden_dense=32, num_directions=1, batch_first=True, min_delta=1e-2,
                 device='cpu'):
        super(RNN_model, self).__init__()
        """
        Args:
        - input_size: The dimensionality of the input data
        - output_size: The dimensionality of the output data
        - n_hidden: The size of the hidden layer, i.e. the number of hidden units used
        - n_layers: The number of hidden layers
        - model_type: The type of modle used ("lstm"/"gru"/"rnn")
        - lr: Learning rate used for training
        - num_epochs: The number of epochs used for training
        - num_directions: Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
        - batch_first: Option to have batches with the batch dimension as the starting dimension 
        of the input data
        """
        # Defining some parameters
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size

        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        # Predefined:
        ## Use only the forward direction
        self.num_directions = num_directions

        ## The input tensors must have shape (batch_size,...)
        self.batch_first = batch_first

        # Defining the recurrent layers
        if model_type.lower() == "rnn":  # RNN
            self.rnn = nn.RNN(input_size=self.input_size*2, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "lstm":  # LSTM
            self.rnn = nn.LSTM(input_size=self.input_size*2, hidden_size=self.hidden_dim,
                               num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru":  # GRU
            self.rnn = nn.GRU(input_size=self.input_size*2, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, batch_first=self.batch_first)
        else:
            print("Model type unknown:", model_type.lower())
            sys.exit()

            # Fully connected layer to be used for mapping the output
        # self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)

        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        # Add a dropout layer with 20% probability
        # self.d1 = nn.Dropout(p=0.2)
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size*2, out_channels=self.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Tanh(),
            nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.input_size*2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0

    def forward(self, y, x_hat):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = y.shape[0]

        # if y.shape[1] != x_hat.shape[1]:
        #     rnn_in = torch.cat((y, x_hat[:, :-1, :]), dim=2)  # (batch_size, 100, 2*input_size)
        # else:
        #     rnn_in = torch.cat((y, x_hat), dim=2)
        rnn_in = torch.cat((y-x_hat, x_hat), dim=2)
        att_weights = self.attention(rnn_in.permute(0, 2, 1)).permute(0, 2, 1)
        # print(rnn_in.shape, att_weights.shape)
        rnn_in = rnn_in * att_weights

        # Obtain the RNN output
        r_out, _ = self.rnn(rnn_in)

        # Reshaping the output appropriately
        r_out_all_steps = r_out.contiguous().view(batch_size, -1, self.num_directions * self.hidden_dim)

        # Passing the output to one fully connected layer
        r_out_final = F.relu(self.fc(r_out_all_steps))

        # Means and variances are computed for time instants t=2, ..., T+1 using the available sequence
        mu_2T_1 = self.fc_mean(r_out_final)  # A second linear projection to get the means
        vars_2T_1 = F.softplus(
            self.fc_vars(r_out_final))  # A second linear projection followed by an activation function to get variances

        # print(mu_2T_1.shape, vars_2T_1.shape)

        # The mean and variances at the first time step need to be computed only based on the previous hidden state
        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(batch_size, 1, -1)
        var_1 = F.softplus(self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))).view(batch_size, 1, -1))

        # print(mu_1.shape, var_1.shape)

        # To get the means and variances for the time instants t=1, ..., T, we take the previous result and concatenate
        # all but last value to the value found at t=1. Concatenation is done along the sequence dimension
        mu = torch.cat(
            (mu_1, mu_2T_1[:, :-1, :]),
            dim=1
        )

        vars = torch.cat(
            (var_1, vars_2T_1[:, :-1, :]),
            dim=1
        )

        return mu, vars

class RnnTwins(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, n_layers, model_type, lr, num_epochs, n_hidden_dense=32,
                 num_directions=1, batch_first = True, min_delta=1e-2, device='cpu'):
        super(RnnTwins, self).__init__()
        self.rnn_f = RNN_model(input_size, output_size, n_hidden, n_layers, model_type, lr, num_epochs, n_hidden_dense,
                               num_directions, batch_first, min_delta, device)
        self.rnn_b = RNN_model(input_size, output_size, n_hidden, n_layers, model_type, lr, num_epochs, n_hidden_dense,
                               num_directions, batch_first, min_delta, device)
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size

        self.model_type = model_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device



    def forward(self, y, x_hat):
        y_f = y
        y_b = torch.flip(y, [1])
        x_hat_f = x_hat
        x_hat_b = torch.flip(x_hat, [1])
        mu_f, vars_f = self.rnn_f(y_f, x_hat_f)
        mu_b, vars_b = self.rnn_b(y_b, x_hat_b)
        mu_b, vars_b = torch.flip(mu_b, [1]), torch.flip(vars_b, [1])
        return mu_f, vars_f, mu_b, vars_b


if __name__ == "__main__":
    y_test = torch.randn((32, 100, 3))
    x_test = torch.randn((32, 100, 3), requires_grad=False)
    # model = RNN_model(input_size=3, output_size=3, n_hidden=32, n_layers=2, model_type='gru', lr=0.001, num_epochs=10, device='cpu')
    model = RnnTwins(input_size=3, output_size=3, n_hidden=32, n_layers=2, model_type='gru', lr=0.001, num_epochs=10, device='cpu')
    mu_f, vars_f, mu_b, vars_b = model.forward(y_test, x_test)
    def combine_covariance_seq(self, m_f, m_b, L_f, L_b):
        # m_f, m_b: batch_size, sequence length, n_obs
        # L_f, L_b: batch_size, seq_length, n_obs, n_obs
        L_combined = []
        for i in range(m_f.shape[1]):
            m_f_i = m_f[:, i, :].unsqueeze(-1)
            m_b_i = m_b[:, i, :].unsqueeze(-1)
            m_i = 0.5 * (m_f_i + m_b_i)
            L_i = 0.5 * (L_f[:, i, :, :] + torch.matmul(m_f_i, m_b_i.transpose(1, 2))) + \
                  0.5 * (L_b[:, i, :, :] + torch.matmul(m_f_i, m_b_i.transpose(1, 2))) - \
                  torch.matmul(m_i, m_i.transpose(1, 2))
            L_combined.append(L_i)
        # return torch.stack(L_combined, dim=1)
        return L_f
    # print(mu_f.shape, vars_f.shape, mu_b.shape, vars_b.shape)
    L_f = torch.diag_embed(vars_f)
    L_b = torch.diag_embed(vars_b)



