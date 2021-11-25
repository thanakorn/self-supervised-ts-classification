import torch
from torch.nn import Module, RNN, LSTM, Linear


class SeqenceEncoder(Module):
    def __init__(self, input_dim: int, hidden_dims: list, cell_type = RNN):
        super(SeqenceEncoder, self).__init__()
        for i, hidden_dim in enumerate(hidden_dims):
            cell = cell_type(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            setattr(self, f'cell{i+1}', cell)
            input_dim = hidden_dim

    def forward(self, x):
        hidden = None
        for cell in self.children():
            if type(cell) == LSTM:
                x, (hidden, _) = cell(x)
            else:
                x, hidden = cell(x)
        
        return hidden[-1]