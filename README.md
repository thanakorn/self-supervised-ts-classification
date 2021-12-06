# Self-supervised Time-series Classification #

This project demonstrates how to tackle time-series classification problem in the situation that there is no label given. The detailed explanation of the method is in this [Medium article](https://medium.com/geekculture/time-series-classification-without-labels-4c3acc5cfd0f).

## Overview ##

The project consists of 4 Python scripts representing each stage of the method as follow:
1. `train_autoencoder.py` : Training a variational autoencoder to extract a compact representation of time-series.
2. `cluster.py` : Assigning clusters to samples in the training set.
3. `pseudo_label.py` : Labeling every training sample by mapping the class that was assigned(manually) to its cluster centroid.
4. `train_classifier.py` : Training a classifier using the label generated py `pseudo_label.py`

All scripts are configured in `config.cfg`.

## How to use the code ##

1. `train_autoencoder.py`
   - Change the `path` under the `data` section of the config file to you data file.
   - Note that the file has to be in a CSV format where each line represents a single time-series and observations are separated by comma.
   - Under the `model` section of the config, adjust the structure of the autoencoder as you wanted.
     - Available parameters are as follow:
       - `cell_type` : RNN, LSTM, or GRU
       - `input_dim` : dimension of input features 
       - `hidden_dims` : dimensions of hidden layers(multiple layers are supported, each layer is separated by comma, f.e. 128,32 -> 2 hidden layers with size 128 and 32 respectively)
       - `embedding_dim` : dimension of representation vector
       - `len` : time-series length
   - Under the `training` section of the config, adjust the hyper-parameters as you wanted.
   - The results are
     - `model.pth` : weights of the model
     - `output.png` : sample outputs
     - `embedding.png` : visualization of embedding vectors(after applying TSNE)

2. `cluster.py`
   - Under the `cluster` section of the config, adjust the numbers of cluster as you wanted.
   - The results are 
     - `cluster.txt` : indicating a cluster id of each sample.
     - `centroids.png` : illustration of each cluster centroid(for manual labeling)
   - After running this script, create a file named `cluster_class.txt`. In this file, assigned a class to each cluster(line 1 = class of cluster 1, line 2 = class of cluster 2, and so on).

3. `pseudo_label.py`
   - The results are
     - `labels.txt` : label of each sample
     - `classes.png` : samples of time-series belonging to each class

4.  `train_classifier.py`
   - Under the `classifier` section of the config, adjust the parameters of classifier as you wanted.
   - The results are
     - `classifier.pkl` : the model in Python pickle format
     - `classifier_output.png` : sample results of trained classifier
     - `classifier_performance.png` : confusion matrix