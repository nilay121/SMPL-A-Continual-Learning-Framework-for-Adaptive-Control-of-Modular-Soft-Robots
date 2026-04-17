import torch
import torch.nn as nn


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out)
#         return out

class LSTMModel_FM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel_FM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.activation = nn.ReLU()
        ## Implement batch norm in a proper way
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
    def forward(self, x):
        out, _ = self.lstm(x)
        out1 = self.fc1(out)
        out1 = out1.permute(0,2,1)
        out = self.batch_norm1(out1)
        out = self.activation(out)
        out = out.permute(0,2,1)
        out2 = self.fc2(out)
        out2 = out2.permute(0,2,1)
        out = self.batch_norm2(out2)
        out = out.permute(0,2,1)
        out = self.activation(out)
        out = self.fc3(out)
        return out
