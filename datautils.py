import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import streamlit as st
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn import datasets
import trimap
from umap.umap_ import UMAP
import pacmap
import seaborn as sns
import matplotlib.pyplot as plt


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

        if hasattr(dataset, 'data'):
            dataset_size = dataset.data.shape[0]
        else:
            dataset_size = dataset.shape[0]

        st.success(f"The full dataset contains {dataset_size} rows.")
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
    data = fetch_20newsgroups(subset='all', return_X_y=True)
    df = pd.DataFrame(data[0])
    df['target'] = data[1]
    return df


def load_lfw_dataset():
    st.write("Loading Labeled Faces in the Wild (LFW) dataset...")
    st.write("It might take a few minutes...")
    data = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    num_samples, num_features = data.data.shape
    feature_names = [f"feature_{i}" for i in range(num_features)]
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
    if not isinstance(dataset, np.ndarray):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()
        else:
            raise ValueError("Dataset must be a numpy array.")

    if dataset.ndim != 2:
        raise ValueError("Dataset must be a 2D array.")

    try:
        tsne = TSNE(n_components=2, **params)
        result = tsne.fit_transform(dataset)

        if result is None or result.size == 0:
            raise ValueError("t-SNE result is empty.")

        return result

    except ValueError as ve:
        print("Value error during t-SNE:", ve)
        return None

    except Exception as e:
        print("Unexpected error during t-SNE:", e)
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


def visualize_results(results):
    if not results:
        st.error("No results to visualize.")
        return

    for technique, data in results.items():
        if data is None or data.size == 0:
            st.error(f"Data for {technique} is empty or null.")
            continue

        if len(data.shape) != 2:
            st.error(f"Data for {technique} must be a 2D array.")
            continue

        plt.figure(figsize=(10, 6))
        try:
            hue = st.session_state['data']['target'] if 'target' in st.session_state['data'] else None
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=hue, palette='tab10', s=50)
            plt.title(f"{technique} Visualization")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(True)
            st.pyplot(plt)
        except ValueError as ve:
            st.error(f"Value error during visualization: {ve}")
        except Exception as e:
            st.error(f"Unexpected error during visualization: {e}")


# def perform_t_sne(dataset, n_components, perplexity, learning_rate, metric):
#     """Perform t-SNE with specified parameters."""
#     try:
#         tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
#         return tsne.fit_transform(dataset)
#     except Exception as e:
#         return f"Error performing t-SNE: {e}"
#
#
# def perform_umap(dataset, n_neighbors=15, min_dist=0.1):
#     """Perform UMAP with specified parameters."""
#     try:
#         umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
#         return umap_model.fit_transform(dataset)
#     except Exception as e:
#         return f"Error performing UMAP: {e}"
#
#
# def perform_trimap(dataset, n_neighbors=15):
#     """Perform TRIMAP with specified parameters."""
#     try:
#         trimap_model = trimap.TRIMAP(n_inliers=n_neighbors)
#         return trimap_model.fit_transform(dataset)
#     except Exception as e:
#         return f"Error performing TRIMAP: {e}"
#
#
# def perform_pacmap(dataset, n_components=2, n_neighbors=15):
#     """Perform PaCMAP with specified parameters."""
#     try:
#         pac_map = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
#         return pac_map.fit_transform(dataset)
#     except Exception as e:
#         return f"Error performing PaCMAP: {e}"
