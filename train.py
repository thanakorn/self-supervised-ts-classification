import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from configparser import ConfigParser
from dataloader.ecg_dataset import ECGDataset
from model.sequence_vae import SequenceVAE
from utils.utils import *

torch.manual_seed(0)
np.random.seed(0)

def compute_loss(loss_fn, input, output):
    recon, mean, logvar = output
    recon_loss = loss_fn(recon, input)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    return loss

def main():
    conf = ConfigParser()
    conf.read('config.cfg')

    data = pd.read_csv(conf['data']['path'], delimiter='  ', header=None)
    X = np.expand_dims(data.values[:,1:], axis=-1)
    dataset = ECGDataset(X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    vae = SequenceVAE.from_config(conf._sections['model'])

    train_conf = conf['train']
    optimizer = get_optimizer(train_conf.get('optimizer'))(vae.parameters(), lr=train_conf.getfloat('lr'))
    loss_fn = get_loss_fn(train_conf.get('loss_fn'))()
    epochs = train_conf.getint('epochs')
    for ep in range(epochs):
        ep_loss = 0
        for x in tqdm(dataloader, desc=f'Epoch {ep + 1}'):
            optimizer.zero_grad()
            output = vae(x)
            loss = compute_loss(loss_fn, x, output)
            ep_loss = ep_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm = 5)
            optimizer.step()

    torch.save(vae.state_dict(), 'model.pth')
    
    # Generate sample output
    vae.eval()
    sample_indices = np.random.permutation(X.shape[0]).tolist()[:5]
    samples = torch.tensor(X[sample_indices]).float()
    samples_out = vae(samples).detach().numpy()
    _, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
    for i, ax in enumerate(axes):
        axes[i][0].plot(samples[i])
        axes[i][1].plot(samples_out[i], color='red')

        if i == 0:
            axes[i][0].set_title('Input')
            axes[i][1].set_title('Output')
    plt.savefig('output.png', bbox_inches='tight')

    # Visualize embedding vectors
    embeddings = vae.encoder(torch.tensor(X).float()).detach().numpy()
    dim_reductor = TSNE(n_components=2, perplexity=80, n_iter=3000)
    r_embeddings = dim_reductor.fit_transform(embeddings)
    plt.figure()
    plt.scatter(r_embeddings[:,0], r_embeddings[:,1], marker='.')
    plt.savefig('embeddings.png', bbox_inches='tight')

if __name__=='__main__':
    main()