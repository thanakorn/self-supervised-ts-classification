import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser

def main():

    conf = ConfigParser()
    conf.read('config.cfg')

    cluster_ids = np.loadtxt('cluster.txt').astype(int)
    cluster_class = np.loadtxt('cluster_class.txt').astype(int).tolist()
    cluster_class = {i : c for i, c in enumerate(cluster_class)}
    cluster_to_class = lambda c : cluster_class[c]
    cluster_to_class = np.vectorize(cluster_to_class)
    classes = cluster_to_class(cluster_ids)
    np.savetxt('labels.txt', classes, fmt='%d')

    data = pd.read_csv(conf['data']['path'], delimiter='  ', header=None)
    data = np.expand_dims(data.values[:,1:], axis=-1)
    n_classes = len(np.unique(classes))
    _, axes = plt.subplots(nrows=n_classes, ncols=4, figsize=(8,7))
    
    for c, row in zip(np.unique(classes), axes):
        class_data = data[classes == c]
        for i, col in enumerate(row):
            if i == 0:
                col.set_title('Class %d' % c)
            col.plot(class_data[np.random.choice(len(class_data))])
            col.xaxis.set_visible(False)
            col.yaxis.set_visible(False)
    plt.savefig('classes.png', bbox_inches='tight')
    

if __name__=='__main__':
    main()