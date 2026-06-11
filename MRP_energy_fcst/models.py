import torch
import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # use build-in pytorch RNN model
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)         
        last_step = rnn_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc(out) 
        return out


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)       
        last_step = lstm_out[:, -1, :] 
        out = self.dropout(last_step)
        out = self.fc(out)      
        return out
