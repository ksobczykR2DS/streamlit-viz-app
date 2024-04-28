from matplotlib import pyplot as plt
import seaborn as sns
from datautils import *
import streamlit as st

st.set_page_config(page_title="Multi-Page App", page_icon=":memo:")

# Page functions
def load_page1():
    st.title("Dimensionality Reduction")
    st.write("""
        Interactive app designed for advanced data visualization using techniques like t-SNE, UMAP, TRIMAP, and PaCMAP.
        It supports data loading, sampling, dynamic visualization, and quality metrics assessment.
    """)

    # Wybór datasetu
    dataset_names = ['MNIST Handwritten Digits', '20 Newsgroups Text Data', 'Labeled Faces in the Wild (LFW)', "Upload Dataset"]
    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names)

    # Suwak do wyboru wielkości próbkowania
    sample_percentage = st.slider(
        "Sample Size (in percentage)",
        min_value=1,
        max_value=100,
        value=100
    )

    # Obsługa przesyłania plików przez użytkownika
    if selected_dataset == "Upload Dataset":
        st.write("Drag and drop a file (CSV or Excel) to upload, or choose from disk:")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

        if uploaded_file and st.button("Load Dataset", key="load_predefined_dataset1"):
            try:
                dataset = upload_file(uploaded_file, sample_percentage)
                if isinstance(dataset, str):
                    raise ValueError(dataset)

                st.success(f"The full dataset contains {dataset.shape[0]} rows.")


                if isinstance(dataset, np.ndarray):
                    dataset = pd.DataFrame(dataset)

                sampled_data = dataset.sample(frac=sample_percentage / 100, random_state=42)

                st.session_state['data'] = sampled_data
                st.session_state['dataset_loaded'] = True
                st.success(f"Sample loaded successfully! Sample size: {sampled_data.shape[0]} rows.")

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    # Obsługa predefiniowanych datasetów
    elif selected_dataset in ["MNIST Handwritten Digits", "20 Newsgroups Text Data", "Labeled Faces in the Wild (LFW)"]:
        if st.button("Load Dataset", key="load_predefined_dataset2"):
            try:
                if selected_dataset == 'MNIST Handwritten Digits':
                    dataset = load_mnist_dataset()
                elif selected_dataset == '20 Newsgroups Text Data':
                    dataset = load_20_newsgroups_dataset()
                elif selected_dataset == 'Labeled Faces in the Wild (LFW)':
                    dataset = load_lfw_dataset()

                if isinstance(dataset, np.ndarray):
                    dataset = pd.DataFrame(dataset)

                if hasattr(dataset, 'data'):
                    dataset_size = dataset.data.shape[0]
                else:
                    dataset_size = dataset.shape[0]

                st.success(f"The full dataset contains {dataset_size} rows.")


                if hasattr(dataset, 'data'):
                    sampled_data = pd.DataFrame(dataset.data).sample(frac=sample_percentage / 100, random_state=42)
                    sampled_data['target'] = dataset.target.sample(frac=sample_percentage / 100, random_state=42).values
                else:
                    sampled_data = dataset.sample(frac=sample_percentage / 100, random_state=42)

                st.session_state['data'] = sampled_data
                st.session_state['dataset_loaded'] = True
                st.success(f"Sample loaded successfully! Sample size: {sampled_data.shape[0]} rows.")

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

def load_page2():
    st.title("Choose Technique and Parameters")

    if 'dataset_loaded' in st.session_state and st.session_state['dataset_loaded']:
        technique = st.selectbox("Select Reduction Technique", ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"])

        if technique == "t-SNE":
            n_components = st.slider("Number of components", 2, 3, 2)
            perplexity = st.slider("Perplexity", 5, 50, 30)
            learning_rate = st.slider("Learning Rate", 10, 200, 200)
            metric = st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])

        elif technique == "UMAP":
            n_neighbors = st.slider("Number of Neighbors", 2, 100, 15)
            min_dist = st.slider("Minimum Distance", 0.0, 0.99, 0.1)

        elif technique == "TRIMAP":
            n_neighbors = st.slider("Number of Neighbors", 2, 100, 10)

        elif technique == "PaCMAP":
            n_components = st.slider("Number of Components", 2, 3, 2)
            n_neighbors = st.slider("Number of Neighbors", 2, 100, 10)

        if st.button("Confirm and Run Technique"):
            try:
                data = st.session_state.get('data')

                if data is None or data.empty:
                    raise ValueError("Data is not loaded or empty.")

                if technique == "t-SNE":
                    reduced_data = perform_t_sne(data, n_components, perplexity, learning_rate, metric)
                elif technique == "UMAP":
                    reduced_data = perform_umap(data, n_neighbors, min_dist)
                elif technique == "TRIMAP":
                    reduced_data = perform_trimap(data, n_neighbors)
                elif technique == "PaCMAP":
                    reduced_data = perform_pacmap(data, n_components, n_neighbors)

                st.session_state['reduced_data'] = reduced_data

                # Jeśli istnieje kolumna 'target' w danych, użyj jej jako 'hue'
                labels = None
                if 'target' in st.session_state['data']:
                    labels = st.session_state['data']['target']

                plt.figure(figsize=(10, 6))
                if labels is not None:
                    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='tab10', s=50)
                    plt.title("Dimensionality Reduction with Labels")
                else:
                    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], s=50)
                    plt.title("Dimensionality Reduction Visualization")

                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True)
                st.pyplot(plt)

                st.success("Dimensionality reduction and visualization completed!")

            except Exception as e:
                st.error(f"Error performing dimensionality reduction: {e}")

    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")

def load_page3():
    st.title("Page 3")
    st.write("This is the content of page 3.")

def load_page4():
    st.title("Page 4")
    st.write("This is the content of page")


# Operation Management Functions
def select_page(page_name):
    st.session_state.page = page_name


# SIDEBAR
if 'page' not in st.session_state:
    st.session_state.page = "Load Dataset"

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        width: 100%;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* CSS selector for the sidebar title */
    .css-1d391kg {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.markdown("<h1 style='text-align: center;'>Navigation Menu</h1>", unsafe_allow_html=True)

st.sidebar.button("Load Dataset", on_click=select_page, args=("Load Dataset",))
st.sidebar.button("Techniques Set Up", on_click=select_page, args=("Techniques Set Up",))
st.sidebar.button("View Data", on_click=select_page, args=("View Data",))
st.sidebar.button("Experiments", on_click=select_page, args=("Experiments",))

if st.session_state.page == "Load Dataset":
    load_page1()
elif st.session_state.page == "Techniques Set Up":
    load_page2()
elif st.session_state.page == "View Data":
    load_page3()
elif st.session_state.page == "Experiments":
    load_page4()
