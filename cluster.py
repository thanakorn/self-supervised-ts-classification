import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model.sequence_vae import SequenceVAE
from configparser import ConfigParser

def main():
    conf = ConfigParser()
    conf.read('config.cfg')

    data = pd.read_csv(conf['data']['path'], delimiter='  ', header=None)
    X = np.expand_dims(data.values[:,1:], axis=-1)

    vae = SequenceVAE.from_config(conf._sections['model'])
    vae.load_state_dict(torch.load('model.pth'))
    vae.eval()

    embeddings = vae.encoder(torch.tensor(X).float())
    embeddings = vae.laten_encoder(embeddings).detach().numpy()
    n_clusters = int(conf['cluster']['n_clusters'])
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
    kmeans.fit_transform(embeddings)
    dim_reductor = TSNE(n_components=2, perplexity=80, n_iter=3000)
    r_embeddings = dim_reductor.fit_transform(embeddings)

    plt.figure()
    plt.scatter(r_embeddings[:,0], r_embeddings[:,1], marker='.', c=kmeans.labels_)
    plt.savefig('clusters.png', bbox_inches='tight')
    np.savetxt('cluster.txt', kmeans.labels_, fmt='%d')

    centroids = kmeans.cluster_centers_
    ts_centroids = vae.decoder(torch.tensor(centroids).float()).detach().numpy()
    nrows, ncols = int(np.round(np.sqrt(kmeans.n_clusters))), int(np.ceil(np.sqrt(kmeans.n_clusters)))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,6))
    for i in range(n_clusters):
        axes[int(i / ncols)][i % ncols].plot(ts_centroids[i,:,0])
        axes[int(i / ncols)][i % ncols].set_title('Cluster %d' % (i + 1))
        axes[int(i / ncols)][i % ncols].get_xaxis().set_visible(False)

    plt.savefig('centroids.png', bbox_inches='tight')

if __name__=='__main__':
    main()