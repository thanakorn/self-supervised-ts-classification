import torch
from torch.nn import Module, RNN, LSTM, Linear, init
from model.sequence_encoder import SeqenceEncoder
from model.sequence_decoder import SeqenceDecoder
from utils.utils import get_class

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

class SequenceVAE(Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dims: list, cell_type = RNN):
        super(SequenceVAE, self).__init__()
        self.encoder = SeqenceEncoder(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            cell_type=cell_type
        )
        self.laten_encoder = LatentEncoder(hidden_dims[-1], hidden_dims[-1])
        self.decoder = SeqenceDecoder(
            input_dim=hidden_dims[-1], 
            output_dim=input_dim, 
            output_len=seq_len, 
            hidden_dims=hidden_dims[::-1], 
            cell_type=cell_type
        )

    @classmethod
    def from_config(cls, config: dict):
        return SequenceVAE(
                input_dim=int(config['input_dim']), 
                seq_len=int(config['len']), 
                hidden_dims=[int(c) for c in config['hidden_dims'].split(',')], 
                cell_type=get_class('torch.nn', config['cell_type'])
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