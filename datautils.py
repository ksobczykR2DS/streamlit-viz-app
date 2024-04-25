# Python 3.12
import pandas as pd
import numpy as np
import trimap
import umap
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from keras.datasets import cifar10
import os
from umap import UMAP
from trimap import TRIMAP
import pacmap



# Ładowanie i przetwarzanie danych
def load_dataset(name):
    """Load a dataset by name with error handling."""
    try:
        if name == "MNIST":
            data = fetch_openml('mnist_784', version=1)
        elif name == "Fashion-MNIST":
            data = fetch_openml('Fashion-MNIST')
        elif name == "CIFAR-10":
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
            data = {
                'data': np.concatenate((train_images, test_images)).reshape(-1, 32 * 32 * 3),
                'target': np.concatenate((train_labels, test_labels)).flatten()
            }
        else:
            raise ValueError("Invalid dataset name")

        # Walidacja danych
        if 'target' not in data:
            raise ValueError("The dataset must contain a 'target' column.")

        return data

    except Exception as e:
        return f"Error loading dataset: {e}"


def upload_file(uploaded_file):
    """Load a file uploaded by the user with error handling."""
    if uploaded_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == '.csv':
                data = pd.read_csv(uploaded_file, encoding='latin-1')
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Invalid file format. Please upload a CSV or Excel file.")

            if 'target' not in data.columns:
                raise ValueError("The dataset must contain a 'target' column.")

            return data

        except Exception as e:
            return f"Error loading data: {e}"

    return None


def perform_t_sne(dataset, n_components, perplexity, learning_rate, metric):
    """Perform t-SNE with specified parameters."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
    return tsne.fit_transform(dataset)


def perform_umap(dataset, n_neighbors=15, min_dist=0.1):
    """Perform UMAP with specified parameters."""
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    return umap_model.fit_transform(dataset)


def perform_trimap(dataset, n_neighbors=15):
    """Perform TRIMAP with specified parameters."""
    trimap_model = trimap.TRIMAP(n_inliers=n_neighbors)
    return trimap_model.fit_transform(dataset)


def perform_pacmap(dataset, n_components=2, n_neighbors=15):
    """Perform PaCMAP with specified parameters."""
    pac_map = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
    return pac_map.fit_transform(dataset)
