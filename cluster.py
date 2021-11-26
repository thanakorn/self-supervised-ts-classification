import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from model.sequence_vae import SequenceVAE
from configparser import ConfigParser

def main():
    conf = ConfigParser()
    conf.read('config.cfg')

    data = pd.read_csv(conf['data']['path'], delimiter='  ', header=None)
    X = np.expand_dims(data.values[:,1:], axis=-1)

    vae = SequenceVAE.from_config(conf._sections['model'])
    vae.load_state_dict(torch.load('vae.cfg'))
    vae.eval()

    embeddings = vae.encoder(torch.tensor(X).float()).detach().numpy()
    n_clusters = int(conf['cluster']['n_clusters'])
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    kmeans.fit_transform(embeddings)
    centroids = kmeans.cluster_centers_
    ts_centroids = vae.decoder(torch.tensor(centroids).float()).detach().numpy()
    nrows, ncols = int(np.sqrt(kmeans.n_clusters)), int(np.ceil(np.sqrt(kmeans.n_clusters)))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))
    for i in range(n_clusters):
        axes[int(i / ncols)][i % ncols].plot(ts_centroids[i,:,0])
        axes[int(i / ncols)][i % ncols].set_title('Cluster %d' % (i + 1))

    plt.savefig('centroids.png', bbox_inches='tight')

if __name__=='__main__':
    main()