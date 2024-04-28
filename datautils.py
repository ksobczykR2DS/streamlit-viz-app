import pandas as pd
import numpy as np
import trimap
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from keras.datasets import cifar10
import os
import pacmap
import streamlit as st
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn import datasets


def load_mnist_dataset():
    st.write("Loading MNIST Handwritten Digits dataset...")
    st.write("It might take a few minutes...")
    data = fetch_openml(name='mnist_784', version=1)
    return data


def load_20_newsgroups_dataset():
    st.write("Loading 20 Newsgroups Text Data dataset...")
    st.write("It might take a few minutes...")
    data = fetch_20newsgroups(subset='all', return_X_y=True)  # Zwraca tuple (data, target)
    df = pd.DataFrame(data[0])  # Konwertuj do DataFrame
    df['target'] = data[1]  # Dodaj kolumnę 'target'
    return df


def load_lfw_dataset():
    st.write("Loading Labeled Faces in the Wild (LFW) dataset...")
    st.write("It might take a few minutes...")
    data = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    num_samples, num_features = data.data.shape
    feature_names = [f"feature_{i}" for i in range(num_features)]  # Generate column names
    df = pd.DataFrame(data.data, columns=feature_names)
    return df


# Funkcja do ładowania zestawu danych
def load_dataset(name, sample_percentage=100):
    """Load a dataset by name with error handling."""
    try:
        if name == "MNIST":
            data = fetch_openml('mnist_784', as_frame=True)
        elif name == "Fashion-MNIST":
            data = fetch_openml('Fashion-MNIST', as_frame=True)
        elif name == "CIFAR-10":
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
            data = {
                'data': np.concatenate((train_images, test_images)).reshape(-1, 32 * 32 * 3),
                'target': np.concatenate((train_labels, test_labels)).flatten()
            }
        elif name == "20 Newsgroups":
            data = fetch_20newsgroups(subset='all', as_frame=True)
        elif name == "LFW":
            data = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
            data = {
                'data': data.images.reshape(-1, data.images.shape[-1]),
                'target': data.target.tolist()
            }
        else:
            raise ValueError("Invalid dataset name")

        if not isinstance(data, pd.DataFrame):
            if 'data' in data and 'target' in data:
                df = pd.DataFrame(data['data'])
                df['target'] = data['target']
                data = df
            else:
                raise ValueError("Invalid dataset structure")

        # Próbkowanie
        sample_data = data.sample(frac=sample_percentage / 100, random_state=42)

        return sample_data
    except Exception as e:
        return f"Error loading dataset: {e}"

# Funkcja do ładowania plików przez użytkownika
def upload_file(uploaded_file, sample_percentage=100):
    if uploaded_file:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == '.csv':
                data = pd.read_csv(uploaded_file, encoding='latin-1')
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Invalid file format. Please upload a CSV or Excel file.")

            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data is not in DataFrame format")

            sample_data = data.sample(frac=sample_percentage / 100, random_state=42)

            return sample_data
        except Exception as e:
            return f"Error loading data: {e}"

    return None


# Funkcje redukcji wymiarów
def run_t_sne(dataset, **params):
    print("Running t-SNE with params:", params)
    try:
        if isinstance(dataset, np.ndarray) and dataset.ndim == 2:
            print("Dataset shape:", dataset.shape)
            tsne = TSNE(n_components=2, **params)
            result = tsne.fit_transform(dataset)
            return result
        else:
            raise ValueError("Dataset must be a 2D numpy array.")
    except Exception as e:
        print("Error performing t-SNE:", e)
        return None


def run_umap(dataset, n_neighbors, min_dist, metric):
    try:
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        return umap.fit_transform(dataset)
    except Exception as e:
        return f"Error performing UMAP: {e}"


def run_trimap(dataset, n_inliers, n_outliers, n_random, weight_adj, n_iters):
    try:
        trimap_transformer = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers,
                                           n_random=n_random, weight_adj=weight_adj, n_iters=n_iters)
        return trimap_transformer.fit_transform(dataset)
    except Exception as e:
        return f"Error performing TRIMAP: {e}"


def run_pacmap(dataset, n_neighbors, mn_ratio, fp_ratio):
    try:
        pacmap_transformer = pacmap.PaCMAP(n_neighbors=n_neighbors, MN_ratio=mn_ratio, FP_ratio=fp_ratio)
        return pacmap_transformer.fit_transform(dataset)
    except Exception as e:
        return f"Error performing PaCMAP: {e}"
