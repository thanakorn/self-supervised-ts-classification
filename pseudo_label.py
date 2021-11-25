import torch
import pandas as pd
import numpy as np
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
    kmeans = KMeans(n_clusters=int(conf['cluster']['n_clusters']), max_iter=1000)
    kmeans.fit_transform(embeddings)
    np.save('labels.npy', np.array(kmeans.labels_))

if __name__=='__main__':
    main()