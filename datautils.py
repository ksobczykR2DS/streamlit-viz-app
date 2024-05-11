import time
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn import datasets
import trimap
from umap.umap_ import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets as torch_datasets, transforms
from torch.utils.data import DataLoader
from pacmap import PaCMAP
import openml
import plotly.express as px



# Ostatnia kolumna musi być targetem, dane tylko numeryczne prócz ostatniej kolumny (jest git)
def handle_uploaded_file(uploaded_file, sample_percentage):
    df = pd.read_csv(uploaded_file)
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)

    if df.iloc[:, :-1].select_dtypes(include=[np.number]).shape[1] != df.shape[1] - 1:
        raise ValueError("All columns except the last must be numeric.")

    sampled_df = df.sample(frac=sample_percentage / 100, random_state=42).reset_index(drop=True)
    sampled_data = sampled_df.iloc[:, :-1]
    sampled_labels = sampled_df.iloc[:, -1]

    st.session_state['data'] = sampled_data
    st.session_state['labels'] = sampled_labels
    st.session_state['dataset_loaded'] = True
    st.experimental_rerun()

# dekompozycja jesli nie zwariujemy, teoretycznie powinno działać dla kazdego ID z openml
def get_dataset_from_openml(dataset_id, sample_percentage):
    with st.spinner('Downloading dataset...'):
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True
        )

    progress_bar = st.progress(0)
    x, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    progress_bar.progress(25)

    x_df = pd.DataFrame(x, columns=attribute_names)
    if 'target' in attribute_names:
        y_series = pd.Series(y, name='target')
    else:
        y_series = pd.Series(y, name='class')

    if 0 < sample_percentage <= 100:
        sample_fraction = sample_percentage / 100.0
        sampled_df = x_df.sample(frac=sample_fraction, random_state=42)
        sampled_labels = y_series[sampled_df.index]
        progress_bar.progress(75)

        st.session_state['data'] = sampled_df
        st.session_state['labels'] = sampled_labels
        st.session_state['dataset_loaded'] = True
        progress_bar.progress(100)
    else:
        raise ValueError("Sample percentage must be between 0 and 100")

    return sampled_df

def get_dating_data(sample_percentage):
    dating_dataset_id = 1130
    return get_dataset_from_openml(dating_dataset_id, sample_percentage)

# 40926 - cifar 10 działa
# 312 - scene działa
def get_scene_data(sample_percentage):
    scene_dataset_id = 312
    return get_dataset_from_openml(scene_dataset_id, sample_percentage)


def get_sign_mnist_data(sample_percentage):
    sign_mnist_dataset_id = 45082
    return get_dataset_from_openml(sign_mnist_dataset_id, sample_percentage)

def get_cifar_10_data(sample_percentage):
    cifar_10_dataset_id = 40926
    return get_dataset_from_openml(cifar_10_dataset_id, sample_percentage)

def dataset_sampling(dataset, sample_percentage):
    try:
        if isinstance(dataset, np.ndarray):
            dataset = pd.DataFrame(dataset)
        elif hasattr(dataset, 'data') and hasattr(dataset, 'target'):
            dataset = pd.DataFrame(dataset.data, columns=[f"feature_{i}" for i in range(dataset.data.shape[1])])
            dataset['target'] = dataset.target
        if not isinstance(dataset, pd.DataFrame) or dataset.empty:
            raise ValueError("Dataset format is not supported or empty.")

        sampled_df = dataset.sample(frac=sample_percentage / 100, random_state=42).reset_index(drop=True)

        st.session_state['data'] = sampled_df.iloc[:, :-1]
        st.session_state['labels'] = sampled_df.iloc[:, -1]
        st.session_state['dataset_loaded'] = True
        st.success(f"Sampled data contains {sampled_df.shape[0]} rows and {sampled_df.shape[1]} columns.")

    except ValueError as ve:
        st.error(f"Value error: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# te funkcje do sprawdzenia wszystkiego git ale jak juz dopracujemy predefined datasets to wywalić bo nic przez nie \
# nie widze
def handle_predefined_datasets(selected_dataset, sample_percentage):
    try:
        if selected_dataset == 'SignMNIST Dataset':
            get_sign_mnist_data(sample_percentage)
        elif selected_dataset == 'Scene Dataset':
            get_scene_data(sample_percentage)
        elif selected_dataset == 'Dating Dataset':
            get_dating_data(sample_percentage)
        elif selected_dataset == 'CIFAR-10 Dataset':
            get_cifar_10_data(sample_percentage)
        else:
            st.error("Dataset not recognized. Please select a valid dataset.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


# konwersja do np array dla przyśpieszenia (zostaje)
def convert_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        return dataset.values
    elif not isinstance(dataset, np.ndarray):
        raise ValueError("Dataset must be either a pandas DataFrame or a numpy array.")
    return dataset


# przydałoby sie unormować te funckej bo w t-sne jest **params, a w reszcie wymienione (lepiej chyba wszędzie **params)
def run_t_sne(dataset, perplexity=30, early_exaggeration=12, learning_rate=200, n_iter=300, metric='euclidean'):
    with st.spinner('Running t-SNE...'):
        try:
            dataset = convert_dataset(dataset)
            if dataset is None or dataset.size == 0:
                st.error('Converted dataset is empty or None.')
                return None

            tsne = TSNE(
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                metric=metric
            )
            result = tsne.fit_transform(dataset)
            if result is None or len(result) == 0:
                st.error('TSNE transformation returned None or empty results.')
            return result
        except Exception as e:
            st.error(f"An error occurred while running t-SNE: {str(e)}")
            return None


def run_umap(dataset, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    with st.spinner('Running UMAP...'):
        try:
            dataset = convert_dataset(dataset)
            if dataset is None or dataset.size == 0:
                st.error('Converted dataset is empty or None.')
                return None

            umap = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric
            )
            result = umap.fit_transform(dataset)
            if result is None or len(result) == 0:
                st.error('UMAP transformation returned None or empty results.')
            return result
        except Exception as e:
            st.error(f"An error occurred while running UMAP: {str(e)}")
            return None

def run_trimap(dataset, n_inliers=10, n_outliers=5, n_random=5, weight_adj=500, n_iters=300):
    with st.spinner('Running TRIMAP...'):
        try:
            dataset = convert_dataset(dataset)
            if dataset is None or dataset.size == 0:
                st.error('Converted dataset is empty or None.')
                return None

            trimap_transformer = trimap.TRIMAP(
                n_inliers=n_inliers,
                n_outliers=n_outliers,
                n_random=n_random,
                weight_adj=weight_adj,
                n_iters=n_iters
            )
            result = trimap_transformer.fit_transform(dataset)
            if result is None or len(result) == 0:
                st.error('TRIMAP transformation returned None or empty results.')
            return result
        except Exception as e:
            st.error(f"An error occurred while running TRIMAP: {str(e)}")
            return None

def run_pacmap(dataset, n_neighbors=50, mn_ratio=0.5, fp_ratio=2.0):
    with st.spinner('Running PaCMAP...'):
        try:
            dataset = convert_dataset(dataset)
            if dataset is None or dataset.size == 0:
                st.error('Converted dataset is empty or None.')
                return None

            pacmap_instance = PaCMAP(
                n_neighbors=n_neighbors,
                MN_ratio=mn_ratio,
                FP_ratio=fp_ratio
            )
            result = pacmap_instance.fit_transform(dataset)
            if result is None or len(result) == 0:
                st.error('PaCMAP transformation returned None or empty results.')
            return result
        except Exception as e:
            st.error(f"An error occurred while running PaCMAP: {str(e)}")
            return None



# przydałaby się dekompozycja (ale moze zostać)
def visualize_individual_result(data, result, labels, title="Result Visualization"):
    result_df = pd.DataFrame(result, columns=['Component 1', 'Component 2'])
    result_df['Label'] = labels

    fig = px.scatter(result_df, x='Component 1', y='Component 2', color='Label', title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

    st.plotly_chart(fig, use_container_width=True)


def main():
    try:
        sample_data = get_dataset_from_openml(1476, 10)  # Gas sensors dataset with 10% sample
        print(sample_data.head())  # Print the first few rows to inspect
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
