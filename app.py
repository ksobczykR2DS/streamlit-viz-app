from utils.datautils import *
from utils.run_experiments_random import compute_cf_nn, compute_cf, perform_experiments
from utils.PCA_analysis import *

st.set_page_config(page_title="Multi-Page App", page_icon=":memo:")


def load_page1():
    st.title("Explore Data Dimensions: Interactive Visualization")
    st.write("""
        Interactive app designed for advanced data visualization using techniques like t-SNE, UMAP, TRIMAP, and PaCMAP.
        Ideal for researchers and data scientists seeking in-depth analysis and visual insights.
    """)

    dataset_names = [
        'MNIST Handwritten Digits',
        'Fashion-MNIST Clothing Items',
        'Natural Scene Images',
        'Your Custom Dataset'
    ]

    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names, index=0)
    uploaded_file = None
    if selected_dataset == "Your Custom Dataset":
        st.write("""
                        **Data Upload Guidelines:**
                        1) Data must be numerical, only labels can be strings (and it is recommended they are \
                        for better visualization readability).
                        2) The last column should be the target variable for classification problems.
                        3) Please ensure your file is in CSV, XLSX, or XLS format.
                    """)
        uploaded_file = st.file_uploader("Upload your dataset file", type=["csv", "xlsx", "xls"])

    sample_percentage = st.slider(
        "Sample Size (in percentage)",
        min_value=1,
        max_value=100,
        value=100
    )

    if st.button("Load Dataset", key='load_selected_dataset'):
        if selected_dataset == "Your Custom Dataset" and uploaded_file:
            load_uploaded_data(uploaded_file, sample_percentage)
        elif selected_dataset != "Your Custom Dataset":
            load_other_datasets(selected_dataset, sample_percentage)
        else:
            st.error("Please upload a file or select a predefined dataset.")

    if 'data' in st.session_state and st.session_state.get('dataset_loaded', False):
        st.subheader("Preview of loaded data")
        st.dataframe(st.session_state['data'].head())

        if st.checkbox("Show distribution of a feature", value=False):
            selected_column = st.selectbox('Select column', st.session_state['data'].columns)
            plot_distribution(selected_column)

    st.write("Press 'R' to reset the application if something goes wrong.")


def load_page2():
    st.markdown("""
        <style>
        .info-text {
            padding: 10px;
            font-size: 16px;
            background-color: #f9f9f9;
            border-left: 5px solid #4CAF50;
            margin: 10px 0px;
            border-radius: 5px;
        }
        .params-header {
            margin-top: 15px;
            font-weight: bold;
        }
        .loaded-dataset {
            font-size: 24px;
            text-align: center;
            color: #4CAF50;
            padding: 10px 0;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Choose Technique and Parameters")

    if 'uploaded_file_name' in st.session_state and st.session_state['uploaded_file_name']:
        uploaded_file_name = st.session_state['uploaded_file_name']
        st.markdown(f"<div class='loaded-dataset'>Loaded File: {uploaded_file_name}</div>", unsafe_allow_html=True)
    elif 'dataset_name' in st.session_state and st.session_state['dataset_name']:
        dataset_name = st.session_state['dataset_name']
        st.markdown(f"<div class='loaded-dataset'>Loaded Dataset: {dataset_name}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='loaded-dataset'>No dataset or file uploaded</div>", unsafe_allow_html=True)

    params = {}
    techniques = []

    use_t_sne = st.checkbox("Activate t-SNE")
    if use_t_sne:
        with st.container():
            st.markdown("""
            **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a powerful machine learning algorithm primarily used for exploring high-dimensional data and reducing it to two or three dimensions for visualization. It effectively reveals structures at many different scales, crucial for interpreting complex datasets.
            """, unsafe_allow_html=True)
            params['t_sne'] = {
                "perplexity": st.slider("Perplexity", 5, 100, 30),
                "early_exaggeration": st.slider("Early Exaggeration", 5, 25, 12),
                "learning_rate": st.slider("Learning Rate", 10, 1000, value=200, step=10),
                "n_iter": st.slider("Number of Iterations", 50, 1200, 300),
                "metric": st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
            }
            techniques.append('t-SNE')

    use_umap = st.checkbox("Activate UMAP")
    if use_umap:
        with st.container():
            st.markdown("""
            **UMAP (Uniform Manifold Approximation and Projection)** is similar to t-SNE but often faster and better at preserving the global structure of data, making it useful for classification, clustering, and visualization.
            """, unsafe_allow_html=True)
            params['umap'] = {
                "n_neighbors": st.slider("Number of Neighbors", 10, 200, 15),
                "min_dist": st.slider("Minimum Distance", 0.0, 0.99, 0.1),
                "metric": st.selectbox("Metric (UMAP)", ["euclidean", "manhattan", "cosine"])
            }
            techniques.append('UMAP')

    use_trimap = st.checkbox("Activate TRIMAP")
    if use_trimap:
        with st.container():
            st.markdown("""
            **TRIMAP (Triplet-based Manifold Learning)** is a dimensionality reduction technique that uses triplet constraints to effectively maintain the global geometry of the data. It applies a weight adjustment mechanism to balance distances to closer neighbors and more distant points, preserving the true manifold structure better over large datasets.
            """, unsafe_allow_html=True)
            params['trimap'] = {
                "n_inliers": st.slider("Number of Inliers", 2, 100, 10),
                "n_outliers": st.slider("Number of Outliers", 1, 50, 5),
                "n_random": st.slider("Number of Random", 1, 50, 5),
                "weight_adj": st.slider("Weight Adjustment", 100, 1000, 500),
                "n_iters": st.slider("Number of Iterations (TRIMAP)", 50, 1200, 300)
            }
            techniques.append('TRIMAP')

    use_pacmap = st.checkbox("Activate PaCMAP")
    if use_pacmap:
        with st.container():
            st.markdown("""
            **PaCMAP (Pairwise Controlled Manifold Approximation Projection)** focuses on the pairwise relationships and controlled organization of nearest neighbors. It is exceptional at preserving both local and global data structures, making it highly suitable for detailed exploratory data analysis.
            """, unsafe_allow_html=True)
            params['pacmap'] = {
                "n_neighbors": st.slider("Number of Neighbors (PaCMAP)", 10, 200, 50),
                "mn_ratio": st.slider("MN Ratio", 0.1, 1.0, 0.5, 0.1),
                "fp_ratio": st.slider("FP Ratio", 1.0, 5.0, 2.0, 0.1)
            }
            techniques.append('PaCMAP')

    if st.button("Confirm and Run Techniques"):
        results = {}
        cf_scores = {}
        st.write(f"Selected Techniques: {techniques}")
        for technique in techniques:
            if technique == 't-SNE':
                result = run_t_sne(st.session_state['data'], **params['t_sne'])
            elif technique == 'UMAP':
                result = run_umap(st.session_state['data'], **params['umap'])
            elif technique == 'TRIMAP':
                result = run_trimap(st.session_state['data'], **params['trimap'])
            elif technique == 'PaCMAP':
                result = run_pacmap(st.session_state['data'], **params['pacmap'])

            results[technique] = result
            if result is not None:
                st.session_state[f'{technique}_result'] = result
                visualize_individual_result(data=st.session_state['data'], result=result, labels=st.session_state['labels'], title=f'{technique} Result')
                cf_nn_values = compute_cf_nn(result, st.session_state['labels'])
                cf_scores[technique] = compute_cf(cf_nn_values)
                st.write(f"{technique} CF Score: {cf_scores[technique]:.4f}")

        st.session_state['reduced_data'] = results
        st.success("Selected techniques executed successfully.")

    if st.button("Reset"):
        keys_to_remove = ['reduced_data', 't-SNE_result', 'UMAP_result', 'TRIMAP_result', 'PaCMAP_result']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()


def load_page3():
    st.title("PCA Components Analysis")
    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.error("No dataset loaded or dataset is empty. Please load a dataset first.")
        return

    if 'downloaded' not in st.session_state:
        st.session_state['downloaded'] = False

    if 'analysis_performed' not in st.session_state:
        st.session_state['analysis_performed'] = False

    if st.checkbox('Select specific features for PCA'):
        selected_features = st.multiselect('Select features', st.session_state['data'].columns)
        data_for_pca = st.session_state['data'][selected_features] if selected_features else st.session_state['data']
    else:
        data_for_pca = st.session_state['data']

    max_components = min(len(data_for_pca.columns), len(data_for_pca))
    n_components = st.slider("Number of Principal Components", min_value=2, max_value=max_components,
                             value=min(3, max_components))

    pca_options = {
        "Standard PCA": None,
        "Kernel PCA (poly)": "poly",
        "Kernel PCA (rbf)": "rbf",
        "Kernel PCA (sigmoid)": "sigmoid",
        "Kernel PCA (cosine)": "cosine"
    }
    pca_choice = st.selectbox("Choose PCA Type", options=list(pca_options.keys()))
    pca_type = "Kernel PCA" if "Kernel PCA" in pca_choice else "Standard PCA"
    kernel = pca_options[pca_choice]

    run_pca = st.checkbox("📊 Show PCA Plot", value=False)
    run_explained_variance = st.checkbox("📈 Show Explained Variance Plot", value=False)
    run_loadings_heatmap = st.checkbox("🔥 Show Loadings Heatmap", value=False)

    if st.button("Run Selected PCA Analyses"):
        with st.spinner(f"Performing {pca_type} with {n_components} components..."):
            if pca_type == "Kernel PCA":
                components, variance_ratio = perform_kernel_pca(data_for_pca, n_components, kernel)
            else:
                components, variance_ratio = perform_pca(data_for_pca, n_components)

            st.session_state['components'] = components
            st.session_state['variance_ratio'] = variance_ratio
            st.session_state['analysis_performed'] = True
            st.success("PCA analysis completed!")

            if run_pca:
                if n_components == 2:
                    plot_pca(components, labels=st.session_state.get('labels'))
                elif n_components > 2:
                    plot_pca_3d(components, labels=st.session_state.get('labels'))

            if run_explained_variance:
                plot_explained_variance(variance_ratio)

            if run_loadings_heatmap:
                pca = PCA(n_components=min(10, len(data_for_pca.columns)))
                pca.fit(data_for_pca)
                plot_pca_loadings_heatmap(pca, data_for_pca.columns)

    col1, col2 = st.columns(2)
    with col1:
        if 'components' in st.session_state:
            components_df = pd.DataFrame(st.session_state['components'])
            csv_data = export_analysis(components_df)
            download_button = st.download_button(
                label="Download PCA Analysis as CSV",
                data=csv_data,
                file_name="pca_analysis.csv",
                mime='text/csv'
            )

    with col2:
        if st.session_state.get('analysis_performed', False):
            if st.button('Reset Analysis'):
                keys_to_clear = ['components', 'variance_ratio', 'analysis_performed', 'downloaded']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()

    if st.session_state.get('downloaded', False):
        st.success("File downloaded successfully!")


def load_page4():
    st.title("Experiments")
    st.write("Run experiments to optimize dimensionality reduction techniques based on the CF score and visualize the results.")

    if 'data' not in st.session_state or 'labels' not in st.session_state:
        st.error("Please load your dataset and labels first.")
        return

    dataset = np.array(st.session_state['data'], dtype='float64')
    labels = st.session_state['labels']

    techniques_list = ['t-SNE', 'UMAP']
    selected_technique = st.selectbox('Select a technique to include in the experiments:', techniques_list)
    n_iter = st.slider("Select number of iterations for optimization:", min_value=10, max_value=100, value=50, step=10)
    verbose = st.checkbox("Show detailed information.")

    if st.button('Run Experiments'):
        st.write("Running experiments...")
        try:
            results = perform_experiments(dataset, labels, [selected_technique], n_iter, verbose)
            if results:
                st.write("Experiments completed successfully.")
                best_params = {}
                for result in results:
                    st.subheader(f"Results for {result['Model']} random search:")
                    st.write(f"Best parameters to perform {result['Model']}:")
                    for param, value in result.items():
                        if param not in ['Model', 'Score', 'estimator']:
                            best_params[param] = value
                            if isinstance(value, float):
                                st.write(f"\t{param}: {value:.2f}")
                            else:
                                st.write(f"\t{param}: {value}")

                results_df = pd.DataFrame([best_params])
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download experiment results as CSV",
                    data=csv,
                    file_name='experiment_results.csv',
                    mime='text/csv',
                )

            else:
                st.write("No results to display.")
        except Exception as e:
            st.error(f"An error occurred while running experiments: {str(e)}")


# ----- sidebar and page management


def select_page(page_name):
    st.session_state.page = page_name


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
st.sidebar.button("Techniques Set Up and Visualization", on_click=select_page, args=("Techniques Set Up and Visualization",))
st.sidebar.button("PCA Components Analysis", on_click=select_page, args=("PCA Components Analysis",))
st.sidebar.button("Technique Tuning with RandomSearch", on_click=select_page, args=("Technique Tuning with RandomSearch",))

if st.session_state.page == "Load Dataset":
    load_page1()
elif st.session_state.page == "Techniques Set Up and Visualization":
    load_page2()
elif st.session_state.page == "PCA Components Analysis":
    load_page3()
elif st.session_state.page == "Technique Tuning with RandomSearch":
    load_page4()
