import pandas as pd
import numpy as np
import trimap
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from keras.datasets import cifar10, mnist
import os
import pacmap
import streamlit as st
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn import datasets


def create_synthetic_data(n_samples=300, n_features=50, n_clusters=3):
    """
    Generates a synthetic dataset using a mixture of Gaussian distributions.

    Parameters:
    - n_samples : int, default 300
        Total number of samples to generate.
    - n_features : int, default 50
        Number of features for each sample.
    - n_clusters : int, default 3
        Number of distinct clusters.

    Returns:
    - data : numpy.ndarray
        The generated dataset as a 2D numpy array (n_samples, n_features).
    - labels : numpy.ndarray
        The integer labels corresponding to cluster membership of each sample.
    """
    np.random.seed(42)  # For reproducibility
    data = []
    labels = []
    samples_per_cluster = n_samples // n_clusters

    for i in range(n_clusters):
        # Generate random mean and covariance
        mean = np.random.rand(n_features) * 100
        cov = np.eye(n_features) * np.random.rand(n_features)  # Diagonal covariance

        # Generate samples for the cluster
        cluster_data = np.random.multivariate_normal(mean, cov, samples_per_cluster)
        data.append(cluster_data)
        labels += [i] * samples_per_cluster

    # Concatenate all cluster data and labels
    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels


def handle_uploaded_file(uploaded_file, sample_percentage):
    try:
        dataset = upload_file(uploaded_file, sample_percentage)
        if isinstance(dataset, str):
            raise ValueError(dataset)
        st.success(f"The full dataset contains {dataset.shape[0]} rows.")
        dataset_sampling(dataset, sample_percentage)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def handle_predefined_datasets(selected_dataset, sample_percentage):
    try:
        if selected_dataset == 'MNIST Handwritten Digits':
            dataset = load_mnist_dataset()
        elif selected_dataset == '20 Newsgroups Text Data':
            dataset = load_20_newsgroups_dataset()
        elif selected_dataset == 'Labeled Faces in the Wild (LFW)':
            dataset = load_lfw_dataset()

        st.success(f"The full dataset contains {dataset.shape[0]} rows.")
        dataset_sampling(dataset, sample_percentage)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def dataset_sampling(dataset, sample_percentage):
    if isinstance(dataset, np.ndarray):
        dataset = pd.DataFrame(dataset)

    if hasattr(dataset, 'data'):
        sampled_data = pd.DataFrame(dataset.data).sample(frac=sample_percentage / 100, random_state=42)
        sampled_data['target'] = dataset.target.sample(frac=sample_percentage / 100, random_state=42).values
    else:
        sampled_data = dataset.sample(frac=sample_percentage / 100, random_state=42)
        sampled_data = sampled_data.values

    st.session_state['data'] = sampled_data
    st.session_state['dataset_loaded'] = True
    st.success(f"Sample loaded successfully! Sample size: {sampled_data.shape[0]} rows.")


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
