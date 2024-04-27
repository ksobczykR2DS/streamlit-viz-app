import pandas as pd
import numpy as np
import trimap
import umap
from sklearn.manifold import TSNE
from keras.datasets import cifar10
import os
import pacmap
import streamlit as st
from sklearn.datasets import fetch_openml


def load_mnist_dataset():
    st.write("Loading MNIST Handwritten Digits dataset...")
    data = fetch_openml(name='mnist_784', version=1)
    st.write("Completed loading MNIST Handwritten Digits. Please proceed to next page.")
    return data

def load_20_newsgroups_dataset():
    st.write("Loading 20 Newsgroups Text Data dataset...")
    data = load_20_newsgroups()
    st.write("Completed loading 20 Newsgroups Text. Please proceed to next page.")
    return data


def load_human_activity_dataset():
    st.write("Loading Human Activity Recognition dataset...")
    data = load_human_activity()
    st.write("Completed loading Human Activity Recognition. Please proceed to next page.")
    return data


def load_gene_expression_dataset():
    st.write("Loading Gene Expression Cancer RNA-Seq dataset...")
    data = load_gene_expression()
    st.write("Completed loading Gene Expression Cancer RNA-Seq. Please proceed to next page.")
    return data


def load_breast_imaging_dataset():
    st.write("Loading Curated Breast Imaging Subset dataset...")
    data = load_breast_imaging()
    st.write("Completed loading Curated Breast Imaging Subset. Please proceed to next page.")
    return data


def load_lfw_dataset():
    st.write("Loading Labeled Faces in the Wild (LFW) dataset...")
    data = load_lfw()
    st.write("Completed loading Faces in the Wild. Please proceed to next page.")
    return data


# Funkcja do ładowania zestawu danych
def load_dataset(name):
    """Load a dataset by name with error handling."""
    try:
        if name == "MNIST":
            data = fetch_openml('mnist_784', version=1, as_frame=True)
        elif name == "Fashion-MNIST":
            data = fetch_openml('Fashion-MNIST', as_frame=True)
        elif name == "CIFAR-10":
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
            data = {
                'data': np.concatenate((train_images, test_images)).reshape(-1, 32 * 32 * 3),
                'target': np.concatenate((train_labels, test_labels)).flatten()
            }
        else:
            raise ValueError("Invalid dataset name")

        if 'target' not in data:
            raise ValueError("The dataset must contain a 'target' column.")

        return data
    except Exception as e:
        return f"Error loading dataset: {e}"

# Funkcja do ładowania plików przez użytkownika
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

# Funkcje redukcji wymiarów
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
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
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



