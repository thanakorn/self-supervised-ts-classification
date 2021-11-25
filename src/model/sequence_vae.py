import torch
from torch.nn import Module, RNN, LSTM, Linear, init
from model.sequence_encoder import SeqenceEncoder
from model.sequence_decoder import SeqenceDecoder

class LatentEncoder(Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(LatentEncoder, self).__init__()
        self.hidden_to_mean = Linear(hidden_dim, embedding_dim)
        self.hidden_to_logvar = Linear(hidden_dim, embedding_dim)

        init.xavier_uniform_(self.hidden_to_mean.weight)
        init.xavier_uniform_(self.hidden_to_logvar.weight)
    
    def forward(self, x):
        mean = self.hidden_to_mean(x)
        if self.training:
            logvar = self.hidden_to_logvar(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return (mean + (eps * std), mean, logvar)
        else:
            return mean

class TimeSeriesVAE(Module):
    def __init__(self, input_size: int, seq_len: int, hidden_sizes: list, cell_type = RNN):
        super(TimeSeriesVAE, self).__init__()
        self.encoder = SeqenceEncoder(
            input_size=input_size, 
            hidden_sizes=hidden_sizes, 
            cell_type=cell_type
        )
        self.laten_encoder = LatentEncoder(hidden_sizes[-1], hidden_sizes[-1])
        self.decoder = SeqenceDecoder(
            input_size=hidden_sizes[-1], 
            output_size=input_size, 
            seq_len=seq_len, 
            hidden_sizes=hidden_sizes[::-1], 
            cell_type=cell_type
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.training:
            embedding, mean, logvar = self.laten_encoder(x)
            out = self.decoder(embedding)
            return out, mean, logvar
        else:
            embedding = self.laten_encoder(x)
            out = self.decoder(embedding)
            return out