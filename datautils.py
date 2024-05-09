import time
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn import datasets
import trimap
from umap.umap_ import UMAP
import pacmap
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets as torch_datasets, transforms
from torch.utils.data import DataLoader
import streamlit_ext as ste
from sklearn.datasets import make_classification


def create_synthetic_data(n_samples, n_features, n_clusters):
    data, labels = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_clusters)
    return data, labels


# Ostatnia kolumna musi być targetem, dane tylko numeryczne prócz ostatniej kolumny
def validate_and_separate_data(df):
    if df.iloc[:, :-1].select_dtypes(include=[np.number]).shape[1] != df.shape[1] - 1:
        raise ValueError("All columns except the last must be numeric.")
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return data, labels


def handle_uploaded_file(uploaded_file, sample_percentage):
    df = pd.read_csv(uploaded_file)
    data, labels = validate_and_separate_data(df)
    sampled_data = data.sample(frac=sample_percentage / 100, random_state=42)
    st.session_state['data'] = sampled_data
    st.session_state['labels'] = labels
    st.success("Dataset loaded and validated successfully!")

    if st.button("Generate Synthetic Data", key="load_synthetic_dataset"):
        data, labels = create_synthetic_data(n_samples=1000, n_features=50, n_clusters=3)
        sampled_data, sampled_labels = data.sample(frac=sample_percentage / 100, random_state=42), labels
        st.session_state['data'] = sampled_data
        st.session_state['labels'] = sampled_labels
        st.success("Synthetic data generated and loaded successfully!")


def handle_predefined_datasets(selected_dataset, sample_percentage):
    try:
        if selected_dataset == 'MNIST Handwritten Digits':
            dataset = load_mnist_dataset()
        elif selected_dataset == 'Labeled Faces in the Wild (LFW)':
            dataset = load_lfw_dataset()
        elif selected_dataset == 'CIFAR-100':
            dataset = load_cifar100_dataset()
        elif selected_dataset == 'Fashion-MNIST':
            dataset = load_fashion_mnist_dataset()
        elif selected_dataset == 'EMNIST':
            dataset = load_emnist_dataset()
        elif selected_dataset == 'KMNIST':
            dataset = load_kmnist_dataset()
        elif selected_dataset == 'Street View House Numbers (SVHN)':
            dataset = load_svhn_dataset()
        else:
            st.error("Dataset not recognized. Please select a valid dataset.")
            return

        if isinstance(dataset, pd.DataFrame):
            dataset_size = dataset.shape[0]
        elif hasattr(dataset, 'data'):
            dataset_size = dataset.data.shape[0]
        else:
            dataset_size = len(dataset)

        st.success(f"The full dataset contains {dataset_size} rows.")
        dataset_sampling(dataset, sample_percentage)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def dataset_sampling(dataset, sample_percentage):
    try:
        if isinstance(dataset, np.ndarray):
            dataset = pd.DataFrame(dataset)

        if hasattr(dataset, 'data') and hasattr(dataset, 'target'):
            full_data = pd.DataFrame(dataset.data)
            full_data['target'] = dataset.target
            sampled_data = full_data.sample(frac=sample_percentage / 100, random_state=42)

        elif isinstance(dataset, pd.DataFrame):
            sampled_data = dataset.sample(frac=sample_percentage / 100, random_state=42)
        else:
            raise ValueError("Dataset format is not supported.")

        if sampled_data.empty:
            raise ValueError("Sampled data is empty.")

        st.session_state['data'] = sampled_data.iloc[:, :-1]
        st.session_state['labels'] = sampled_data.iloc[:, -1]
        st.session_state['dataset_loaded'] = True
        st.success(f"Sample loaded successfully! Sample size: {sampled_data.shape[0]} rows.")

    except ValueError as ve:
        st.error(f"Value error: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def load_mnist_dataset():
    st.write("Loading MNIST Handwritten Digits dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    data = fetch_openml(name='mnist_784', version=1)

    for i in range(1, 101):
        time.sleep(0.02)
        progress_bar.progress(i)

    progress_bar.empty()

    return data


def load_lfw_dataset():
    st.write("Loading Labeled Faces in the Wild (LFW) dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    data = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    for i in range(1, 101):
        time.sleep(0.02)
        progress_bar.progress(i)

    num_samples, num_features = data.data.shape
    feature_names = [f"feature_{i}" for i in range(num_features)]

    df = pd.DataFrame(data.data, columns=feature_names)
    progress_bar.empty()
    return df

# Funkcja do ładowania CIFAR-100
def load_cifar100_dataset():
    st.write("Loading CIFAR-100 dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    for i in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(i)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    df = pd.DataFrame(data[0].view(len(dataset), -1).numpy())
    df['target'] = data[1].numpy()

    progress_bar.empty()

    return df


# Funkcja do ładowania Fashion-MNIST
def load_fashion_mnist_dataset():
    st.write("Loading Fashion-MNIST dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    for i in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(i)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    df = pd.DataFrame(data[0].view(len(dataset), -1).numpy())
    df['target'] = data[1].numpy()

    progress_bar.empty()

    return df


# Funkcja do ładowania EMNIST
def load_emnist_dataset():
    st.write("Loading EMNIST dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)

    for i in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(i)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    df = pd.DataFrame(data[0].view(len(dataset), -1).numpy())
    df['target'] = data[1].numpy()

    progress_bar.empty()

    return df


# Funkcja do ładowania KMNIST
def load_kmnist_dataset():
    st.write("Loading KMNIST dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_datasets.KMNIST(root='./data', train=True, download=True, transform=transform)

    for i in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(i)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    df = pd.DataFrame(data[0].view(len(dataset), -1).numpy())
    df['target'] = data[1].numpy()

    progress_bar.empty()

    return df


def load_svhn_dataset():
    st.write("Loading Street View House Numbers (SVHN) dataset...")
    st.write("It might take a few minutes...")

    progress_bar = st.progress(0)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torch_datasets.SVHN(root='./data', split='train', download=True, transform=transform)

    for i in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(i)

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))
    df = pd.DataFrame(data[0].view(len(dataset), -1).numpy())
    df['target'] = data[1].numpy()

    progress_bar.empty()

    return df


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
        if isinstance(dataset, str):
            raise AttributeError("Data is a string. Cannot proceed with TRIMAP.")

        if isinstance(dataset, pd.DataFrame):
            dataset.columns = dataset.columns.astype(str)

        trimap_transformer = trimap.TRIMAP(
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_random=n_random,
            weight_adj=weight_adj,
            n_iters=n_iters
        )
        result = trimap_transformer.fit_transform(dataset)

        return result

    except AttributeError as ae:
        st.error(f"AttributeError: {ae}")
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def visualize_results(results):
    if not results:
        st.error("No results to visualize.")
        return

    for technique, data in results.items():
        # Obsługa błędu związana z 'size'
        if isinstance(data, str):
            st.error("Data is a string. Cannot use 'size' attribute.")
            continue

        if data is None or not hasattr(data, 'size') or data.size == 0:
            st.error(f"Data for {technique} is empty or invalid.")
            continue

        if len(data.shape) != 2:
            st.error(f"Data for {technique} must be a 2D array.")
            continue

        plt.figure(figsize=(10, 6))

        if 'data' not in st.session_state or 'target' not in st.session_state['data']:
            st.warning("No 'target' column found. Visualization will be monochrome.")
            sns.scatterplot(x=data[:, 0], y=data[:, 1], s=50)
        else:
            hue = st.session_state['data']['target']
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=hue, palette='tab10', s=50)

        plt.title(f"{technique} Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(plt)

        plot_path = f"{str(technique)}.png"
        plt.savefig(plot_path)

        with open(plot_path, "rb") as file:
            btn = ste.download_button(
                label="Download image",
                data=file,
                file_name=plot_path,
                mime="image/png"
            )


def perform_t_sne(dataset, n_components, perplexity, learning_rate, metric):
    """Perform t-SNE with specified parameters."""
    try:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
        return tsne.fit_transform(dataset)
    except Exception as e:
        return f"Error performing t-SNE: {e}"


def perform_umap(dataset, n_neighbors=15, min_dist=0.1):
    """Perform UMAP with specified parameters."""
    try:
        umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
        return umap_model.fit_transform(dataset)
    except Exception as e:
        return f"Error performing UMAP: {e}"


def perform_trimap(dataset, n_neighbors=15):
    """Perform TRIMAP with specified parameters."""
    try:
        trimap_model = trimap.TRIMAP(n_inliers=n_neighbors)
        return trimap_model.fit_transform(dataset)
    except Exception as e:
        return f"Error performing TRIMAP: {e}"


def perform_pacmap(dataset, n_components=2, n_neighbors=15):
    """Perform PaCMAP with specified parameters."""
    try:
        pac_map = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
        return pac_map.fit_transform(dataset)
    except Exception as e:
        return f"Error performing PaCMAP: {e}"
