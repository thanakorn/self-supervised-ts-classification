import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.nn import LSTM, L1Loss
from torch.optim import Adam
from dataloader.ecg_dataset import ECGDataset
from model.sequence_vae import SequenceVAE
from tqdm import tqdm
from argparse import ArgumentParser

def compute_loss(loss_fn, input, output):
    recon, mean, logvar = output
    recon_loss = loss_fn(recon, input)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    return loss

def main(args):
    data = pd.read_csv('data/ECG5000_TRAIN.txt', delimiter='  ', header=None)
    X = np.expand_dims(data.values[:,1:], axis=-1)
    Y = data.values[:,0].astype(int)

    dataset = ECGDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    vae = SequenceVAE(input_dim=X.shape[-1], seq_len=X.shape[1], hidden_dims=[64, 32, 16], cell_type=LSTM)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    loss_fn = L1Loss()
    
    epochs = args.epochs
    for ep in range(epochs):
        ep_loss = 0
        for x in tqdm(dataloader, desc=f'Epoch {ep + 1}'):
            optimizer.zero_grad()
            output = vae(x)
            loss = compute_loss(loss_fn, x, output)
            ep_loss = ep_loss + loss.item()
            loss.backward()
            optimizer.step()
    torch.save(vae.state_dict(), 'vae.cfg')
    
    # Generate sample output
    vae.eval()
    sample_indices = np.random.permutation(X.shape[0]).tolist()[:5]
    samples = torch.tensor(X[sample_indices]).float()
    samples_out = vae(samples).detach().numpy()
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
    for i, ax in enumerate(axes):
        axes[i][0].plot(samples[i])
        axes[i][1].plot(samples_out[i], color='red')

        if i == 0:
            axes[i][0].set_title('Input')
            axes[i][1].set_title('Output')
    plt.savefig('output.png', bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)