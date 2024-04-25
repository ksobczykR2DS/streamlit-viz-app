# Python 3.12
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10
import os
from umap import UMAP
from trimap import TRIMAP
import pacmap



# Loading and Sampling Data functions
def load_dataset(name):
    """Load a dataset by name."""
    if name == "MNIST":
        data = fetch_openml('mnist_784', version=1)
    elif name == "Fashion-MNIST":
        data = fetch_openml('Fashion-MNIST')
    elif name == "CIFAR-10":
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        data = {
            'data': np.concatenate((train_images, test_images)).reshape(-1, 3072),
            'target': np.concatenate((train_labels, test_labels)).flatten()
        }
    else:
        return "Invalid dataset name"

    if 'target' not in data:
        return "Dataset must contain a 'target' column"

    return data


def upload_file(uploaded_file):
    """Load a file uploaded by the user."""
    if uploaded_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == '.csv':
                data = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Invalid file format. Please upload a CSV or Excel file.")

            # Walidacja danych
            if 'target' not in data.columns:
                raise ValueError("The dataset must contain a 'target' column")

            return data

        except Exception as e:
            return f"Error loading data: {e}"

    return None


# Perform techniques
# Dla każdego ustawić domyślne wartości parametrów, jeśli użytkownik zmieni to podmianka
# Te funkcje tylko dla implementacji metody i zwrot, odrębne funkcję robiące wizualizację
def perform_t_sne(dataset, n_components, perplexity, learning_rate, metric):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
    return tsne.fit_transform(dataset)

def perform_umap(dataset, n_neighbors=15, min_dist=0.1):
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    return umap.fit_transform(dataset)

def perform_trimap(dataset, n_neighbors=15):
    trimap = TRIMAP(n_inliers=n_neighbors)
    return trimap.fit_transform(dataset)

def perform_pacmap(dataset, n_components=2, n_neighbors=15):
    pac_map = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
    return pac_map.fit_transform(dataset)
