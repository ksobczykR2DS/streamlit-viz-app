import matplotlib.pyplot as plt
import streamlit as st
from datautils import *
from itertools import product
import time

# To run up as website (with venv activated)
# USAGE:  streamlit run app.py
if 'dataset_loaded' not in st.session_state:
    st.session_state['dataset_loaded'] = False

tab1, tab2, tab3, tab4 = st.tabs(["Select Dataset", "Choose Technique and Parameters", "Data Visualization", "Experiments"])

with tab1:
    st.header("Dataset Selection")
    dataset_name = st.selectbox("Choose a dataset or upload your own", ["MNIST", "Fashion-MNIST", "CIFAR-10", "Upload Dataset"])

    if dataset_name == "Upload Dataset":
        uploaded_file = st.file_uploader("Choose a file (CSV or Excel)")
        if uploaded_file:
            try:
                data = upload_file(uploaded_file)
                if isinstance(data, str):
                    raise ValueError(data)

                st.session_state['data'] = data
                st.session_state['dataset_loaded'] = True
                st.success("Dataset successfully loaded!")

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    elif dataset_name in ["MNIST", "Fashion-MNIST", "CIFAR-10"]:
        if st.button("Load Dataset"):
            data = load_dataset(dataset_name)
            if isinstance(data, str):
                st.error(data)
            else:
                st.session_state['data'] = data
                st.session_state['dataset_loaded'] = True
                st.success("Dataset successfully loaded!")

with tab2:
    if st.session_state['dataset_loaded']:
        st.header("Choose Technique and Parameters")

        technique = st.selectbox("Select Reduction Technique", ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"])

        try:
            if technique == "t-SNE":
                n_components = st.slider("Number of components", 2, 3, 2)
                perplexity = st.slider("Perplexity", 5, 50, 30)
                learning_rate = st.slider("Learning Rate", 10, 200, 200)
                metric = st.selectbox("Metric", ["euclidean", "manhattan", "cosine"])

                # Wykonanie t-SNE
                data = st.session_state['data']['data']
                reduced_data = perform_t_sne(data, n_components, perplexity, learning_rate, metric)
                st.session_state['reduced_data'] = reduced_data
                st.success("t-SNE completed!")

            elif technique == "UMAP":
                n_neighbors = st.slider("Number of Neighbors", 2, 100, 15)
                min_dist = st.slider("Minimum Distance", 0.0, 0.99, 0.1)

                data = st.session_state['data']['data']
                reduced_data = perform_umap(data, n_neighbors, min_dist)
                st.session_state['reduced_data'] = reduced_data
                st.success("UMAP completed!")

            elif technique == "TRIMAP":
                n_neighbors = st.slider("Number of Neighbors", 2, 100, 10)

                data = st.session_state['data']['data']
                reduced_data = perform_trimap(data, n_neighbors)
                st.session_state['reduced_data'] = reduced_data
                st.success("TRIMAP completed!")

            elif technique == "PaCMAP":
                n_components = st.slider("Number of Components", 2, 3, 2)
                n_neighbors = st.slider("Number of Neighbors", 2, 100, 10)

                data = st.session_state['data']['data']
                reduced_data = perform_pacmap(data, n_components, n_neighbors)
                st.session_state['reduced_data'] = reduced_data
                st.success("PaCMAP completed!")

        except Exception as e:
            st.error(f"Error performing dimensionality reduction: {e}")

    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")

with tab3:
    if st.session_state['dataset_loaded']:
        st.header("Data Visualization")

        if 'reduced_data' in st.session_state:
            progress_bar = st.progress(0)
            progress_steps = 100

            for step in range(progress_steps):
                time.sleep(0.05)
                progress_bar.progress(step + 1)

            fig, ax = plt.subplots()
            ax.scatter(st.session_state['reduced_data'][:, 0], st.session_state['reduced_data'][:, 1],
                       c=st.session_state['data']['target'], cmap='viridis')
            st.pyplot(fig)

        else:
            st.warning("No data to visualize. Please complete a reduction technique.")

    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")

with tab4:
    if st.session_state['dataset_loaded']:
        st.header("Experiments")

        technique = st.selectbox("Select Reduction Technique for Experimentation", ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"])

        if technique == "t-SNE":
            n_components_range = st.slider("Number of Components", 2, 3, (2, 3))
            perplexity_range = st.slider("Perplexity Range", 5, 50, (5, 50))
            learning_rate_range = st.slider("Learning Rate Range", 10, 200, (10, 200))

            param_combinations = list(product(
                range(n_components_range[0], n_components_range[1] + 1),
                range(perplexity_range[0], perplexity_range[1] + 1),
                range(learning_rate_range[0], learning_rate_range[1] + 10)
            ))

            results = []
            for n_components, perplexity, learning_rate in param_combinations:
                reduced_data = perform_t_sne(
                    st.session_state['data']['data'],
                    n_components,
                    perplexity,
                    learning_rate,
                    "euclidean"
                )
                results.append({
                    "n_components": n_components,
                    "perplexity": perplexity,
                    "learning_rate": learning_rate,
                    "reduced_data": reduced_data
                })

            st.session_state['experiment_results_t_sne'] = results
            st.write("Experiments completed!")

        elif technique == "UMAP":
            n_neighbors_range = st.slider("Number of Neighbors Range", 2, 100, (2, 100))
            min_dist_range = np.linspace(0.1, 0.5, 5)

            param_combinations = list(product(
                range(n_neighbors_range[0], n_neighbors_range[1] + 1),
                min_dist_range
            ))

            results = []
            for n_neighbors, min_dist in param_combinations:
                reduced_data = perform_umap(
                    st.session_state['data']['data'],
                    n_neighbors,
                    min_dist
                )
                results.append({
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "reduced_data": reduced_data
                })

            st.session_state['experiment_results_umap'] = results
            st.write("Experiments completed!")

        elif technique == "TRIMAP":
            n_neighbors_range = st.slider("Number of Neighbors Range", 2, 100, (5, 20))

            param_combinations = list(product(range(n_neighbors_range[0], n_neighbors_range[1] + 1)))

            results = []
            for n_neighbors in param_combinations:
                reduced_data = perform_trimap(
                    st.session_state['data']['data'],
                    n_neighbors[0]
                )
                results.append({
                    "n_neighbors": n_neighbors[0],
                    "reduced_data": reduced_data
                })

            st.session_state['experiment_results_trimap'] = results
            st.write("Experiments completed!")

        elif technique == "PaCMAP":
            n_components_range = st.slider("Number of Components Range", 2, 3, (2, 3))
            n_neighbors_range = st.slider("Number of Neighbors Range", 2, 100, (5, 20))

            param_combinations = list(product(
                range(n_components_range[0], n_components_range[1] + 1),
                range(n_neighbors_range[0], n_neighbors_range[1] + 1)
            ))

            results = []
            for n_components, n_neighbors in param_combinations:
                reduced_data = perform_pacmap(
                    st.session_state['data']['data'],
                    n_components,
                    n_neighbors
                )
                results.append({
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "reduced_data": reduced_data
                })

            st.session_state['experiment_results_pacmap'] = results
            st.write("Experiments completed!")

    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")
