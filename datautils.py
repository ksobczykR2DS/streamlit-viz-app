# Python 3.12
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10
import os


# Loading and Sampling Data functions
def load_dataset(name):
    """Load a dataset by name."""
    if name == "MNIST":
        data = fetch_openml('mnist_784', version=1)
    elif name == "Fashion-MNIST":
        data = fetch_openml('Fashion-MNIST')
    elif name == "CIFAR-10":
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        data = {'data': np.concatenate((train_images, test_images)),
                'target': np.concatenate((train_labels, test_labels)).flatten()}
    else:
        return None
    return data


def upload_file(uploaded_file):
    """Load a file uploaded by the user."""
    if uploaded_file is not None:
        try:
            if os.path.splitext(uploaded_file.name)[1].lower() in ['.csv']:
                data = pd.read_csv(uploaded_file)
            elif os.path.splitext(uploaded_file.name)[1].lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
            else:
                return None
        except Exception as e:
            return f"Error loading data: {e}"
        return data
    return None


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
