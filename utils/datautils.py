import time
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import streamlit as st
import trimap
from umap.umap_ import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from pacmap import PaCMAP
import openml
import plotly.express as px


# PAGE 1 UTILS
def load_uploaded_data(uploaded_file, sample_percentage):
    try:
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        handle_uploaded_file(uploaded_file, sample_percentage)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")


def load_other_datasets(dataset_name, sample_percentage):
    try:
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)

        dataset_functions = {
            'MNIST Handwritten Digits': lambda: get_mnist_dataset(sample_percentage),
            'Fashion-MNIST Clothing Items': lambda: get_fashion_mnist_data(sample_percentage),
            'Natural Scene Images': lambda: get_scene_data(sample_percentage)
        }

        if dataset_name in dataset_functions:
            data, labels = dataset_functions[dataset_name]()
            st.session_state['data'] = data
            st.session_state['labels'] = labels
            st.session_state['dataset_name'] = dataset_name
            st.session_state['dataset_loaded'] = True
    except Exception as e:
        st.error(f"Error loading data: {e}")


def plot_distribution(selected_column):
    fig, ax = plt.subplots()
    sns.histplot(st.session_state['data'][selected_column], kde=True, ax=ax)
    st.pyplot(fig)


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

    st.session_state['uploaded_file_name'] = uploaded_file.name


def get_dataset_from_openml(dataset_id, sample_percentage):
    with st.spinner('Downloading dataset...'):
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True
        )

    x, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    x_df = pd.DataFrame(x, columns=attribute_names)

    if 'target' in attribute_names:
        y_series = pd.Series(y, name='target')
    else:
        y_series = pd.Series(y, name='class')

    if 0 < sample_percentage <= 100:
        sample_fraction = sample_percentage / 100.0
        sampled_df = x_df.sample(frac=sample_fraction, random_state=42)
        sampled_labels = y_series[sampled_df.index]

        sampled_df.reset_index(drop=True, inplace=True)
        sampled_labels.reset_index(drop=True, inplace=True)

        st.session_state['data'] = sampled_df
        st.session_state['labels'] = sampled_labels
        st.session_state['dataset_loaded'] = True

        return sampled_df, sampled_labels
    else:
        raise ValueError("Sample percentage must be between 0 and 100")


def get_scene_data(sample_percentage):
    return get_dataset_from_openml('312', sample_percentage)


def get_fashion_mnist_data(sample_percentage):
    return get_dataset_from_openml('40996', sample_percentage)


def get_mnist_dataset(sample_percentage):
    return get_dataset_from_openml('554', sample_percentage)


# PAGE 2 UTILS
def convert_and_scale_dataset(dataset):
    if isinstance(dataset, pd.DataFrame):
        return dataset.values
    elif not isinstance(dataset, np.ndarray):
        raise ValueError("Dataset must be either a pandas DataFrame or a numpy array.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def run_t_sne(dataset, perplexity=30, early_exaggeration=12, learning_rate=200, n_iter=300, metric='euclidean'):
    with st.spinner('Running t-SNE...'):
        try:
            dataset = convert_and_scale_dataset(dataset)
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
            dataset = convert_and_scale_dataset(dataset)
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
            dataset = convert_and_scale_dataset(dataset)
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
            dataset = convert_and_scale_dataset(dataset)
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


def visualize_individual_result(data, result, labels, title="Result Visualization"):
    result_df = pd.DataFrame(result, columns=['Component 1', 'Component 2'])
    result_df['Label'] = labels

    fig = px.scatter(result_df, x='Component 1', y='Component 2', color='Label', title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

    st.plotly_chart(fig, use_container_width=True)
