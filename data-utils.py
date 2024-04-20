# Python 3.12

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10


MNIST = fetch_openml('mnist_784', version=1)
FMINST = fetch_openml('Fashion-MNIST')
CIFAT = cifar10.load_data()

# Loading and Sampling Data functions


# Perform techniques
# Dla każdego ustawić domyślne wartości parametrów, jeśli użytkownik zmieni to podmianka
# Te funkcje tylko dla implementacji metody i zwrot, odrębne funkcję robiące wizualizację
def perform_t_sne(dataset, n_components, perplexity, learning_rate, metric):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
    return tsne.fit_transform(dataset)


def perform_umap(dataset):
    pass


def perform_trimap(dataset):
    pass


def perform_pacmap(dataset):
    pass
