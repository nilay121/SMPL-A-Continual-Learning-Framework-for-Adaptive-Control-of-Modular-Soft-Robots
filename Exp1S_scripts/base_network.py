import numpy as np
import torch
import torch.nn as nn

"""
Initially should start from this

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
"""

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, h0, c0):
        x = x.to(torch.float32)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to("cuda")
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to("cuda")
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, hn, cn

class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, args=None):
        super(LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.args = args
        self.lstm2_out_array = []

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, h0, c0):
        x = x.to(torch.float32)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to("cuda")
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to("cuda")
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # ## save the lstm2 output for plotting the cosine similarity
        # self.lstm2_out_array.append(out.cpu().detach().numpy())
        # np.save(f"lateral_analysis/lstm2_out_seg{self.args.n_seg}_lat_{self.args.lat_connect}.npy", np.concatenate(self.lstm2_out_array))
        
        return out, hn, cn

class LSTM_lateral(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, lat_connect="false"):
        super(LSTM_lateral, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lat_connect = lat_connect
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,)

    def forward(self, x):
        x = x.to(torch.float32)
        if self.lat_connect == "true":
            out, _ = self.lstm(x)
        else:
            """
            This ensures your noise matches the original 
            lateral output distribution, and you're not confounding scale vs structure.
            """
            mean_lat_out = x.mean()
            std_lat_out = x.std()
            x_new = torch.randn_like(x) * std_lat_out + mean_lat_out
            # x = torch.rand_like(x)
            out, _ = self.lstm(x_new)
            
        return out

class Linear_layer(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(Linear_layer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.args = args
        self.linear_out_array = []

    def forward(self, x, h_n, c_n):
        out = self.linear_layer(x)
        # self.linear_out_array.append(out.cpu().detach().numpy())
        # np.save(f"lateral_analysis/dense_out_seg{self.args.n_seg}_lat_{self.args.lat_connect}.npy", np.concatenate(self.linear_out_array))
        return out, h_n, c_n
