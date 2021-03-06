import torch
from torch.nn import Module, RNN, LSTM, Linear

class SeqenceDecoder(Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, output_len: int, cell_type = RNN):
        super(SeqenceDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len

        for i, hidden_dim in enumerate(hidden_dims):
            cell = cell_type(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            setattr(self, f'cell{i+1}', cell)
            input_dim = hidden_dim
        self.fc = Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.repeat(1, self.output_len)
        x = x.reshape((batch_size, self.output_len, self.input_dim))

        layers = list(self.children())
        cells, fc = layers[:-1], layers[-1]
        for cell in cells:
            if type(cell) == LSTM:
                x, (_, _) = cell(x)
            else:
                x, _ = cell(x)
        out = fc(x)
        return out