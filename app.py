from datautils import *
import streamlit as st
from PCA_analysis import *
from run_experiments import compute_cf_nn, compute_cf, perform_experiments
from model_utils import create_model, get_embeddings, reduce_dimensions, visualize_embeddings, preprocess_data

st.set_page_config(page_title="Multi-Page App", page_icon=":memo:")




def load_page1():
    st.title("Dimensionality Reduction")
    st.write("""
        Interactive app designed for advanced data visualization using techniques like t-SNE, UMAP, TRIMAP, and PaCMAP.
        It supports data loading, sampling, dynamic visualization, and quality metrics assessment.
    """)

    if st.button("Reset Application"):
        keys_to_clear = ['data', 'sampled_data', 'dataset_loaded']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    dataset_names = [
        'SignMNIST Dataset',
        'Scene Dataset',
        'Dating Dataset',
        'CIFAR-10 Dataset',
        'Upload Dataset',
        'Select a dataset...'
    ]

    def load_dataset(name):
        if name == 'SignMNIST Dataset':
            return get_sign_mnist_data(100)
        elif name == 'Scene Dataset':
            return get_scene_data(100)
        elif name == 'Dating Dataset':
            return get_dating_data(100)
        elif name == 'CIFAR-10 Dataset':
            return get_cifar_10_data(100)

    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names, index=len(dataset_names) - 1)

    if selected_dataset and selected_dataset != "Select a dataset...":
        if selected_dataset == "Upload Dataset":
            uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
            if uploaded_file:
                if st.button("Load Dataset", key='load_uploaded_data'):
                    with st.spinner('Loading data...'):
                        handle_uploaded_file(uploaded_file, 100)
                    st.success("Dataset loaded successfully!")
        else:
            if st.button(f"Load {selected_dataset}", key=f'load_{selected_dataset}'):
                with st.spinner(f'Loading {selected_dataset}...'):
                    load_dataset(selected_dataset)
                st.success(f"{selected_dataset} loaded successfully!")

        if 'data' in st.session_state and not st.session_state['data'].empty:
            st.subheader("Preview of loaded data")
            st.dataframe(st.session_state['data'].head())
            if st.checkbox("Show distribution of a feature"):
                selected_column = st.selectbox('Select column', st.session_state['data'].columns)
                fig, ax = plt.subplots()
                sns.histplot(st.session_state['data'][selected_column], kde=True, ax=ax)
                st.pyplot(fig)

            sample_percentage = st.slider(
                "Sample Size (in percentage)",
                min_value=1,
                max_value=100,
                value=100
            )
            if st.button("Sample Data", key='sample_data_button'):
                sampled_data = dataset_sampling(st.session_state['data'], sample_percentage)
                if sampled_data is not None and not sampled_data.empty:
                    st.session_state['sampled_data'] = sampled_data



def load_page2():
    if 'data' not in st.session_state:
        st.error("No dataset loaded or dataset is empty. Please load a dataset first.")
        return

    dataset = st.session_state['data']
    labels = st.session_state['labels']

    st.title("Choose Technique and Parameters")

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
        techniques.append('PaCMAP')
        params['pacmap'] = {
            "n_neighbors": st.slider("Number of Neighbors (PaCMAP)", 10, 200, 15),
            "mn_ratio": st.slider("MN Ratio", 0.1, 1.0, 0.5, 0.1),
            "fp_ratio": st.slider("FP Ratio", 1.0, 5.0, 2.0, 0.1)
        }

    if st.button("Confirm and Run Techniques"):
        results = {}
        cf_scores = {}
        st.write(f"Selected Techniques: {techniques}")
        for technique in techniques:
            if technique == 't-SNE':
                result = run_t_sne(dataset, **params['t_sne'])
            elif technique == 'UMAP':
                result = run_umap(dataset, **params['umap'])
            elif technique == 'TRIMAP':
                result = run_trimap(dataset, **params['trimap'])
            elif technique == 'PaCMAP':
                result = run_pacmap(dataset, **params['pacmap'])

            results[technique] = result
            if result is not None:
                visualize_individual_result(data=dataset, result=result, labels=labels, title=f'{technique} Result')
                cf_nn_values = compute_cf_nn(result, labels)
                cf_scores[technique] = compute_cf(cf_nn_values)
                st.write(f"{technique} CF Score: {cf_scores[technique]:.4f}")

        st.session_state['reduced_data'] = results
        st.success("Selected techniques executed successfully.")


def load_page3():
    st.title("PCA Components Analysis")
    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.error("No dataset loaded or dataset is empty. Please load a dataset first.")
        return

    st.write("## PCA Options")
    n_components = st.selectbox("Number of Principal Components", options=[2, 3], index=0)

    # Inicjalizacja zmiennych dla checkboxów z domyślną wartością False
    run_2d_pca = False
    run_3d_pca = False
    run_biplot = False
    run_explained_variance = False
    run_loadings_heatmap = False

    # Warunkowe wyświetlanie checkboxów
    if n_components == 2:
        run_2d_pca = st.checkbox("📉 Show 2D PCA", value=False)

    if n_components == 3:
        run_3d_pca = st.checkbox("📊 Show 3D PCA", value=False)

    if n_components == 2:
        run_biplot = st.checkbox("🔍 Show Biplot", value=False)

    run_explained_variance = st.checkbox("📈 Show Explained Variance Plot", value=False)
    run_loadings_heatmap = st.checkbox("🔥 Show Loadings Heatmap", value=False)

    if st.button("Run Selected PCA Analyses"):
        with st.spinner(f"Performing PCA with {n_components} components..."):
            components, variance_ratio = perform_pca(st.session_state['data'], n_components)
            st.session_state['components'] = components
            st.session_state['variance_ratio'] = variance_ratio
            st.success("PCA analysis completed!")

        if run_2d_pca and n_components == 2:
            plot_pca(components, labels=st.session_state.get('labels'))
        if run_3d_pca and n_components == 3:
            plot_pca_3d(components, labels=st.session_state.get('labels'))
        if run_biplot and n_components == 2:
            plot_pca_biplot(st.session_state['data'], labels=st.session_state.get('labels'))
        if run_explained_variance:
            plot_explained_variance(variance_ratio)
        if run_loadings_heatmap:
            pca = PCA(n_components=min(10, len(st.session_state['data'].columns)))
            pca.fit(st.session_state['data'])
            plot_pca_loadings_heatmap(pca, st.session_state['data'].columns)


def load_page4():
    # gridsearch, randomsearch
    st.title("Experiments")
    st.write("Run experiments to optimize dimensionality reduction techniques based on the CF score and visualize the results.")

    if 'data' not in st.session_state:
        st.error("Please load your dataset first.")
        return

    dataset = st.session_state.get('data', None)
    labels = st.session_state.get('labels', None)

    techniques_list = ['t-SNE', 'UMAP', 'TRIMAP', 'PaCMAP']
    selected_technique = st.selectbox('Select a technique to include in the experiments:', techniques_list)

    # Number of iterations for the BayesSearchCV
    n_iter = st.slider("Select number of iterations for optimization:", min_value=10, max_value=100, value=50, step=10)

    if st.button('Run Experiments'):
        if selected_technique:
            st.write("Running experiments...")
            try:
                results = perform_experiments(dataset, labels, selected_technique, n_iter)
                if results:
                    st.write("Experiments completed successfully.")
                    for result in results:
                        st.subheader(f"Results for {result['Model']}:")
                        st.write(f"Best CF Score: {result['Score']:.4f}")
                        st.write("Best Parameters:")
                        for param, value in result.items():
                            if param not in ['Model', 'Score']:
                                st.write(f"{param}: {value}")
                else:
                    st.write("No results to display.")
            except Exception as e:
                st.error(f"An error occurred while running experiments: {str(e)}")
        else:
            st.error("Please select a technique to run experiments.")



def load_page5():
    # using plotly
    st.title("3D Embedding Visualization")

    data = st.session_state.get('data', None)
    labels = st.session_state.get('labels', None)
    if data is not None and labels is not None:
        input_shape = data.shape[1] if isinstance(data, pd.DataFrame) else np.array(data).shape[1]
        model = create_model(input_shape)

        if st.button('Generate Embeddings and Visualize'):
            embeddings = get_embeddings(model, data)
            if embeddings is not None:
                reduced_embeddings = reduce_dimensions(embeddings)
                visualize_embeddings(reduced_embeddings, labels)

    else:
        st.warning("No data or labels available. Please load data and labels into session state before proceeding.")


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
st.sidebar.button("PCA Components Analysis", on_click=select_page, args=("PCA Components Analysis",))
st.sidebar.button("Experiments", on_click=select_page, args=("Experiments",))
st.sidebar.button("Interactive Visualization", on_click=select_page, args=("Interactive Visualization",))

if st.session_state.page == "Load Dataset":
    load_page1()
elif st.session_state.page == "Techniques Set Up and Visualization":
    load_page2()
elif st.session_state.page == "PCA Components Analysis":
    load_page3()
elif st.session_state.page == "Experiments":
    load_page4()
elif st.session_state.page == "Interactive Visualization":
    load_page5()
