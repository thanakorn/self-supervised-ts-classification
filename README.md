# Self-supervised Time-series Classification #

This project demonstrates how to tackle time-series classification problem in the situation that there is no label given. The detailed explanation of the method is in this [Medium article](https://medium.com/geekculture/time-series-classification-without-labels-4c3acc5cfd0f).

## Overview ##

The project consists of 4 Python scripts representing each stage of the method as follow:
1. train_autoencoder.py : Training a variational autoencoder to extract a compact representation of time-series.
2. cluster.py : Assigning clusters to samples in the training set.
3. pseudo_label.py : Labeling every training sample by mapping the class that was assigned(manually) to its cluster centroid.
4. train_classifier.py : Training a classifier using the label generated py `pseudo_label.py`

All scripts are configured using `config.cfg`.