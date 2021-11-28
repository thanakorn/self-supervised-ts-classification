import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn
import pickle
from configparser import ConfigParser
from sklearn.ensemble import RandomForestClassifier
from model.sequence_vae import SequenceVAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_cm(cm, ax):
    seaborn.heatmap(cm, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', cbar=False)
    ax.set_title('Train')
    ax.set_xlabel('Predict')
    ax.set_ylabel('Actual')

def main():
    conf = ConfigParser()
    conf.read('config.cfg')

    data = pd.read_csv(conf['data']['path'], delimiter='  ', header=None)
    X = np.expand_dims(data.values[:,1:], axis=-1)
    Y = np.loadtxt('labels.txt').astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    vae = SequenceVAE.from_config(conf._sections['model'])
    vae.load_state_dict(torch.load('model.pth'))
    vae.eval()

    train_embeddings = vae.encoder(torch.tensor(X_train).float())
    train_embeddings = vae.laten_encoder(train_embeddings).detach().numpy()
    
    classifier_conf = conf._sections['classifier']
    classifier = RandomForestClassifier(
        n_estimators=classifier_conf.get('n_estimators', 100),
        max_features=classifier_conf.get('max_features', 'sqrt'),
        max_depth=classifier_conf.get('max_depth', 4),
        random_state=42
    )
    classifier.fit(train_embeddings, Y_train)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    train_predict = classifier.predict(train_embeddings)
    train_cm = confusion_matrix(Y_train, train_predict, normalize='true')

    test_embeddings = vae.encoder(torch.tensor(X_test).float())
    test_embeddings = vae.laten_encoder(test_embeddings).detach().numpy()
    test_predict = classifier.predict(test_embeddings)
    test_cm = confusion_matrix(Y_test, test_predict, normalize='true')

    _, axes = plt.subplots(ncols=2, figsize=(8,4))
    plot_cm(train_cm, axes[0])
    plot_cm(test_cm, axes[1])
    plt.savefig('classifier_performance.png', bbox_inches='tight')

if __name__=='__main__':
    main()