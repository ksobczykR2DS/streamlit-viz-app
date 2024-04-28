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

        # Przycisk do ładowania datasetu
        if st.button("Load Dataset", key="load_predefined_dataset1"):
            if uploaded_file:
                try:
                    dataset = upload_file(uploaded_file, sample_percentage)
                    if isinstance(dataset, str):
                        raise ValueError(dataset)

                    # Komunikat o rozmiarze pełnego datasetu
                    st.success(f"The full dataset contains {dataset.shape[0]} rows.")

                    # Próbkowanie danych
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
    tab1, tab2 = st.tabs(["Technique Selection", "Component Analysis"])

    with tab1:
        st.title("Choose Technique and Parameters")

        results = {}

        if 'data' in st.session_state:
            dataset = st.session_state['data']
            use_t_sne = st.checkbox("Use t-SNE")
            use_umap = st.checkbox("Use UMAP")
            use_trimap = st.checkbox("Use TRIMAP")
            use_pacmap = st.checkbox("Use PaCMAP")

            params = {}

            if use_t_sne:
                st.subheader("t-SNE Parameters")
                params['t_sne'] = {
                    "perplexity": st.slider("Perplexity", 5, 100, 30),
                    "early_exaggeration": st.slider("Early Exaggeration", 5, 25, 12),
                    "learning_rate": st.slider("Learning Rate", 10, 1000, value=200, step=10),
                    "n_iter": st.slider("Number of Iterations", 50, 1200, 300),
                    "metric": st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
                }

            if use_umap:
                st.subheader("UMAP Parameters")
                params['umap'] = {
                    "n_neighbors": st.slider("Number of Neighbors", 10, 200, 15),
                    "min_dist": st.slider("Minimum Distance", 0.0, 0.99, 0.1),
                    "metric": st.selectbox("Metric (UMAP)",
                                           ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra"])
                }

            if use_trimap:
                st.subheader("TRIMAP Parameters")
                params['trimap'] = {
                    "n_inliers": st.slider("Number of Inliers", 2, 100, 10),
                    "n_outliers": st.slider("Number of Outliers", 1, 50, 5),
                    "n_random": st.slider("Number of Random", 1, 50, 5),
                    "weight_adj": st.slider("Weight Adjustment", 100, 1000, 500),
                    "n_iters": st.slider("Number of Iterations (TRIMAP)", 50, 1200, 300)
                }

            if use_pacmap:
                st.subheader("PaCMAP Parameters")
                params['pacmap'] = {
                    "n_neighbors": st.slider("Number of Neighbors (PaCMAP)", 10, 200, 15),
                    "mn_ratio": st.slider("MN Ratio", 0.1, 1.0, 0.5, 0.1),
                    "fp_ratio": st.slider("FP Ratio", 1.0, 5.0, 2.0, 0.1)
                }

            if st.button("Confirm and Run Techniques"):
                if use_t_sne:
                    results['t-SNE'] = run_t_sne(dataset, **params['t_sne'])
                if use_umap:
                    results['UMAP'] = run_umap(dataset, **params['umap'])
                if use_trimap:
                    results['TRIMAP'] = run_trimap(dataset, **params['trimap'])
                if use_pacmap:
                    results['PaCMAP'] = run_pacmap(dataset, **params['pacmap'])

            st.session_state['reduced_data'] = results
            st.success("Selected techniques have been executed successfully.")

        else:
            st.error("Please load a dataset in the 'Load Dataset' tab first.")

        with tab2:
            st.title("Component Analysis")


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
