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

    dataset_names = [
        'MNIST Handwritten Digits',
        'Labeled Faces in the Wild (LFW)',
        'CIFAR-100',
        'Fashion-MNIST',
        'EMNIST',
        'KMNIST',
        'Street View House Numbers (SVHN)',
        "Upload Dataset",
        "Synthetic Data"
    ]

    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names)

    sample_percentage = st.slider(
        "Sample Size (in percentage)",
        min_value=1,
        max_value=100,
        value=100
    )

    if selected_dataset == "Upload Dataset":
        st.write("Drag and drop a file (CSV or Excel) to upload, or choose from disk:")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

        if uploaded_file and st.button("Load Dataset", key="load_predefined_dataset1"):
            handle_uploaded_file(uploaded_file, sample_percentage)

    elif selected_dataset == "Synthetic Data":
        if st.button("Generate Synthetic Data", key="load_synthetic_dataset"):
            data, labels = create_synthetic_data(n_samples=1000, n_features=50, n_clusters=3)
            sampled_data = pd.DataFrame(data).sample(frac=sample_percentage / 100, random_state=42)
            st.session_state['data'] = sampled_data
            st.session_state['dataset_loaded'] = True
            st.success("Synthetic data generated and loaded successfully!")

    else:
        if st.button("Load Dataset", key="load_predefined_dataset2"):
            handle_predefined_datasets(selected_dataset, sample_percentage)


def load_page2():
    if 'data' not in st.session_state:
        st.error("Dataset not loaded in session. Please go back and load a dataset first.")
        return

    dataset = st.session_state.get('data', None)

    if dataset is None or (isinstance(dataset, pd.DataFrame) and dataset.empty):
        st.error("Dataset is not loaded or is empty. Please load a dataset first.")
        return

    tab1, tab2 = st.tabs(["Technique Selection and Visualization", "Component Analysis"])

    with tab1:
        st.title("Choose Technique and Parameters")

        results = {}

        use_t_sne = st.checkbox("Use t-SNE")
        use_umap = st.checkbox("Use UMAP")
        use_trimap = st.checkbox("Use TRIMAP")
        use_pacmap = st.checkbox("Use PaCMAP")

        params = {}
        techniques = []

        if use_t_sne:
            st.subheader("t-SNE Parameters")
            params['t_sne'] = {
                "perplexity": st.slider("Perplexity", 5, 100, 30),
                "early_exaggeration": st.slider("Early Exaggeration", 5, 25, 12),
                "learning_rate": st.slider("Learning Rate", 10, 1000, value=200, step=10),
                "n_iter": st.slider("Number of Iterations", 50, 1200, 300),
                "metric": st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
            }
            techniques.append('t-SNE')

        if use_umap:
            st.subheader("UMAP Parameters")
            params['umap'] = {
                "n_neighbors": st.slider("Number of Neighbors", 10, 200, 15),
                "min_dist": st.slider("Minimum Distance", 0.0, 0.99, 0.1),
                "metric": st.selectbox("Metric (UMAP)",
                                       ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra"])
            }
            techniques.append('UMAP')

        if use_trimap:
            st.subheader("TRIMAP Parameters")
            params['trimap'] = {
                "n_inliers": st.slider("Number of Inliers", 2, 100, 10),
                "n_outliers": st.slider("Number of Outliers", 1, 50, 5),
                "n_random": st.slider("Number of Random", 1, 50, 5),
                "weight_adj": st.slider("Weight Adjustment", 100, 1000, 500),
                "n_iters": st.slider("Number of Iterations (TRIMAP)", 50, 1200, 300)
            }
            techniques.append('TRIMAP')

        if use_pacmap:
            st.subheader("PaCMAP Parameters")
            params['pacmap'] = {
                "n_neighbors": st.slider("Number of Neighbors (PaCMAP)", 10, 200, 15),
                "mn_ratio": st.slider("MN Ratio", 0.1, 1.0, 0.5, 0.1),
                "fp_ratio": st.slider("FP Ratio", 1.0, 5.0, 2.0, 0.1)
            }
            techniques.append('PacMAP')

        if st.button("Confirm and Run Techniques"):
            if dataset is None:
                st.error("Dataset is empty. Cannot run techniques.")
                return

            if not techniques:
                st.error("No techniques selected.")
                return

            # Uruchamianie technik dla wybranego datasetu
            if use_t_sne:
                results['t-SNE'] = run_t_sne(dataset, **params['t_sne'])
            if use_umap:
                results['UMAP'] = run_umap(dataset, **params['umap'])
            if use_trimap:
                results['TRIMAP'] = run_trimap(dataset, **params['trimap'])
            if use_pacmap:
                results['PaCMAP'] = run_pacmap(dataset, **params['pacmap'])

            st.session_state['reduced_data'] = results
            st.success("Selected techniques executed successfully.")

            visualize_results(results)

        with tab2:
            st.title("PCA/Kernel PCA + Component Analysis")


def load_page3():
    st.title("Experiments")
    st.write("This is the content of page")


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
st.sidebar.button("Techniques Set Up", on_click=select_page, args=("Techniques Set Up and Visualization",))
st.sidebar.button("Experiments", on_click=select_page, args=("Experiments",))

if st.session_state.page == "Load Dataset":
    load_page1()
elif st.session_state.page == "Techniques Set Up and Visualization":
    load_page2()
elif st.session_state.page == "Experiments":
    load_page3()
