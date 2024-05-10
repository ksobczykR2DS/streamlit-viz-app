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
    st.success(f"Dataset loaded and validated successfully! Sampled data contains {sampled_data.shape[0]} rows and {sampled_data.shape[1]} columns.")


# dekompozycja jesli nie zwariujemy, teoretycznie powinno działać dla kazdego ID z openml
def get_dataset_from_openml(dataset_id, sample_percentage):
    # Explicitly set download parameters
    # Set these to `True` if you need to immediately download the data and metadata
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True
    )

    x, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # sprawdzenie czy jest kolumna target lub class, jesli nie to bierze ostatnia jako target
    x_df = pd.DataFrame(x, columns=attribute_names)
    if 'target' in x_df.columns or 'Class' in x_df.columns or 'class' in x_df.columns:
        if 'target' in x_df.columns:
            y_df = x_df.pop('target')
        else:
            y_df = x_df.pop('Class')
    else:
        y_df = x_df.pop(x_df.columns[-1])

    # przerzucenie targetu na ostatnia kolumne
    data = pd.concat([x_df, y_df], axis=1)

    if 0 < sample_percentage <= 100:
        sample_fraction = sample_percentage / 100.0
        sampled_df = data.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        sampled_data = sampled_df.iloc[:, :-1]
        sampled_labels = sampled_df.iloc[:, -1]

        st.session_state['data'] = sampled_data
        st.session_state['labels'] = sampled_labels
        st.session_state['dataset_loaded'] = True
        st.success(f"Dataset loaded successfully! Sampled data contains {sampled_data.shape[0]} rows and {sampled_data.shape[1]} columns.")
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


def get_mnist_data(sample_percentage):
    scene_dataset_id = 554
    return get_dataset_from_openml(scene_dataset_id, sample_percentage)


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
        st.success(f"Sample loaded successfully! Sample size: {sampled_df.shape[0]} rows.")

    except ValueError as ve:
        st.error(f"Value error: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# te funkcje do sprawdzenia wszystkiego git ale jak juz dopracujemy predefined datasets to wywalić bo nic przez nie \
# nie widze
def handle_predefined_datasets(selected_dataset, sample_percentage):
    try:
        if selected_dataset == 'MNIST Handwritten Digits':
            get_mnist_data(sample_percentage)
        elif selected_dataset == 'Scene Dataset':
            get_scene_data(sample_percentage)
            return
        elif selected_dataset == 'Dating Dataset':
            get_dating_data(sample_percentage)
            return
        else:
            st.error("Dataset not recognized. Please select a valid dataset.")
            return

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
def run_t_sne(dataset, **params):
    dataset = convert_dataset(dataset)

    tsne = TSNE(**params)
    result = tsne.fit_transform(dataset)

    return result


def run_umap(dataset, n_neighbors, min_dist, metric):
    dataset = convert_dataset(dataset)

    umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )
    result = umap.fit_transform(dataset)

    return result


def run_trimap(dataset, n_inliers, n_outliers, n_random, weight_adj, n_iters):
    dataset = convert_dataset(dataset)

    trimap_transformer = trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        # weight_adj wyrzuca błąd, propozycja zamiany na weight_temp (do sprawdzenia)
        weight_adj=weight_adj,
        n_iters=n_iters
    )
    result = trimap_transformer.fit_transform(dataset)

    return result


def run_pacmap(dataset, n_neighbors, mn_ratio, fp_ratio):
    dataset = convert_dataset(dataset)

    pacmap_instance = PaCMAP(
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio
    )
    result = pacmap_instance.fit_transform(dataset)

    return result


# przydałaby się dekompozycja (ale moze zostać)
def visualize_individual_result(technique, data):
    if data is None or not hasattr(data, 'size') or data.size == 0:
        st.error(f"Data for {technique} is empty or invalid.")
        return

    if len(data.shape) != 2:
        st.error(f"Data for {technique} must be a 2D array.")
        return

    plt.figure(figsize=(10, 6))
    if 'labels' in st.session_state and st.session_state['labels'] is not None:
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=st.session_state['labels'], palette='tab10', s=50)
    else:
        st.warning("No labels found. Visualization will be monochrome.")
        sns.scatterplot(x=data[:, 0], y=data[:, 1], s=50)

    plt.title(f"{technique} Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(plt)

    # it's a feature, not a bug
    plot_path = f"{technique}.png"
    plt.savefig(plot_path)
    with open(plot_path, "rb") as file:
        st.download_button(
            label="Download image",
            data=file,
            file_name=plot_path,
            mime="image/png"
        )


# wszystko poniżej DO WYWALENIA LATER
# ---------------------------------------------------------------------------------------------------------------------
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


def main():
    try:
        sample_data = get_dataset_from_openml(1476, 10)  # Gas sensors dataset with 10% sample
        print(sample_data.head())  # Print the first few rows to inspect
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
