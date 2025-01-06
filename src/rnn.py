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
        model_type, lr, num_epochs, n_hidden_dense=32, num_directions=1, batch_first = True, min_delta=1e-2, device='cpu'):
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
        self.device=device
        
        # Predefined:
        ## Use only the forward direction 
        self.num_directions = num_directions
        
        ## The input tensors must have shape (batch_size,...)
        self.batch_first = batch_first
        
        # Defining the recurrent layers 
        if model_type.lower() == "rnn": # RNN 
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)   
        elif model_type.lower() == "lstm": # LSTM
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)
        elif model_type.lower() == "gru": # GRU
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)  
        else:
            print("Model type unknown:", model_type.lower()) 
            sys.exit() 
        
        # Fully connected layer to be used for mapping the output
        #self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)
        
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, n_hidden_dense).to(self.device)
        self.fc_mean = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        self.fc_vars = nn.Linear(n_hidden_dense, self.output_size).to(self.device)
        # Add a dropout layer with 20% probability
        #self.d1 = nn.Dropout(p=0.2)

    def init_h0(self, batch_size):
        """ This function defines the initial hidden state of the RNN
        """
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        return h0
    
    def forward(self, x):
        """ This function defines the forward function to be used for the RNN model
        """
        batch_size = x.shape[0]
        # x = torch.flip(x, dims=[1])
        # Obtain the RNN output
        r_out, _ = self.rnn(x)
        
        # Reshaping the output appropriately
        r_out_all_steps = r_out.contiguous().view(batch_size, -1, self.num_directions * self.hidden_dim)

        # Passing the output to one fully connected layer
        y = F.relu(self.fc(r_out_all_steps))

        # Means and variances are computed for time instants t=2, ..., T+1 using the available sequence
        mu_2T_1 = self.fc_mean(y) # A second linear projection to get the means
        vars_2T_1 = F.softplus(self.fc_vars(y)) # A second linear projection followed by an activation function to get variances

        # The mean and variances at the first time step need to be computed only based on the previous hidden state
        mu_1 = self.fc_mean(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:]))).view(batch_size, 1, -1)
        var_1 = F.softplus(self.fc_vars(F.relu(self.fc(self.init_h0(batch_size)[-1,:,:]))).view(batch_size, 1, -1))

        # To get the means and variances for the time instants t=1, ..., T, we take the previous result and concatenate 
        # all but last value to the value found at t=1. Concatenation is done along the sequence dimension
        mu = torch.cat(
            (mu_1, mu_2T_1[:,:-1,:]),
            dim=1
        )

        vars = torch.cat(
            (var_1, vars_2T_1[:,:-1,:]),
            dim=1
        )
        # mu = torch.flip(mu, dims=[1])
        # vars = torch.flip(vars, dims=[1])

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

    def forward(self, x):
        x_f = x
        x_b = torch.flip(x, [1])
        mu_f, vars_f = self.rnn_f(x_f)
        mu_b, vars_b = self.rnn_b(x_b)
        mu_b, vars_b = torch.flip(mu_b, [1]), torch.flip(vars_b, [1])
        return mu_f, vars_f, mu_b, vars_b


if __name__ == "__main__":
    import pickle
    data_file = '../data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_sigmae2_-10.0dB_smnr_10.0dB.pkl'
    test_data = pickle.load(open(data_file, 'rb'))
    print(test_data.keys())
    x_test = test_data['X']
    y_test = test_data['Y']

    # y_test = torch.randn((32, 2000, 3), requires_grad=False)
    # x_test = torch.randn((32, 2000, 3), requires_grad=False)
    print(x_test.shape)
    print(y_test.shape)

    # model = RNN_model(input_size=3, output_size=3, n_hidden=32, n_layers=2, model_type='gru', lr=0.001, num_epochs=10, device='cpu')
    model = RnnTwins(input_size=3, output_size=3, n_hidden=32, n_layers=2, model_type='gru', lr=0.001, num_epochs=10, device='cpu')
    mu_f, vars_f, mu_b, vars_b = model.forward(y_test)
    def combine_covariance_seq(m_f, m_b, L_f, L_b):
        # m_f, m_b: batch_size, sequence length, n_obs
        # L_f, L_b: batch_size, seq_length, n_obs, n_obs
        L_combined = []
        for i in range(m_f.shape[1]):
            m_f_i = -10*m_f[:, i, :].unsqueeze(-1)
            m_b_i = m_b[:, i, :].unsqueeze(-1)
            m_i = 0.5 * (m_f_i + m_b_i)

            # print(torch.matmul(m_f_i, m_b_i.transpose(1, 2)))
            # print(torch.matmul(m_i, m_i.transpose(1, 2)))

            L_i = 0.5 * (L_f[:, i, :, :] + torch.matmul(m_f_i, m_f_i.transpose(1, 2))) + \
                  0.5 * (L_b[:, i, :, :] + torch.matmul(m_b_i, m_b_i.transpose(1, 2))) - \
                  torch.matmul(m_i, m_i.transpose(1, 2))
            L_combined.append(L_i)
        return torch.stack(L_combined, dim=1)
        # return L_f

    L_f = torch.diag_embed(vars_f)
    L_b = torch.diag_embed(vars_b)
    # print(mu_f)
    # print(mu_b)
    # print(L_f)
    # print(L_b)

    L = combine_covariance_seq(mu_f, mu_b, L_f, L_b)
    std = np.sqrt(torch.diagonal(torch.squeeze(L[0,:,:,:], 0), offset=0, dim1=1,dim2=2).detach().numpy())
    print(std)