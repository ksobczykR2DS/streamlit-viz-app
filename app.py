import matplotlib.pyplot as plt
import streamlit as st

from build import select_page
from datautils import *
from itertools import product
import time


# To run up as website (with venv activated)
# USAGE:  streamlit run app.py
def select_dataset_page():
    st.title("Select Dataset")
    dataset_name = st.selectbox("Choose a dataset or upload your own", ["MNIST", "Fashion-MNIST", "CIFAR-10", "Upload Dataset"])

    sample_percentage = st.slider(
        "Sample Size (in percentage)",
        min_value=1,
        max_value=100,
        value=100,  # Domyślnie 100%
        key="sample_percentage_slider"
    )  # Domyślnie 100% danych

    if dataset_name == "Upload Dataset":
        st.write("Drag and drop a file (CSV or Excel) to upload, or choose from disk:")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], key="file_uploader")

        if uploaded_file:
            try:
                data = upload_file(uploaded_file, sample_percentage)
                if isinstance(data, str):
                    raise ValueError(data)


                sample_data = data.sample(frac=sample_percentage / 100, random_state=42)

                st.session_state['data'] = sample_data
                st.session_state['dataset_loaded'] = True
                st.success(f"Dataset successfully loaded! Size: {sample_data.shape[0]} rows.")

                if st.button("Proceed to Choose Technique"):
                    st.session_state['page'] = "choose_technique"

            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    elif dataset_name in ["MNIST", "Fashion-MNIST", "CIFAR-10"]:
        if st.button("Load Dataset", key="load_dataset_button"):
            try:
                data = load_dataset(dataset_name, sample_percentage)

                if isinstance(data, str):
                    raise ValueError(data)

                sample_data = data.sample(frac=sample_percentage / 100, random_state=42)
                st.session_state['data'] = sample_data
                st.session_state['dataset_loaded'] = True
                st.success(f"Dataset successfully loaded! Sample size: {sample_data.shape[0]} rows.")

                if st.button("Proceed to Choose Technique", key="proceed_to_technique_2"):
                    st.session_state['page'] = "choose_technique"

            except Exception as e:
                st.error(f"Error loading dataset: {e}")


# Strona do wyboru techniki i parametrów
def choose_technique_page():
    st.title("Choose Technique and Parameters")
    if 'dataset_loaded' in st.session_state and st.session_state['dataset_loaded']:
        if st.button("Back to Dataset Selection"):
            st.session_state['page'] = "select_dataset"

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
                if technique == "t-SNE":
                    data = st.session_state['data']['data']
                    reduced_data = perform_t_sne(data, n_components, perplexity, learning_rate, metric)
                    st.session_state['reduced_data'] = reduced_data
                    st.success("t-SNE completed!")

                elif technique == "UMAP":
                    data = st.session_state['data']['data']
                    reduced_data = perform_umap(data, n_neighbors, min_dist)
                    st.session_state['reduced_data'] = reduced_data
                    st.success("UMAP completed!")

                elif technique == "TRIMAP":
                    data = st.session_state['data']['data']
                    reduced_data = perform_trimap(data, n_neighbors)
                    st.session_state['reduced_data'] = reduced_data
                    st.success("TRIMAP completed!")

                elif technique == "PaCMAP":
                    data = st.session_state['data']['data']
                    reduced_data = perform_pacmap(data, n_components, n_neighbors)
                    st.session_state['reduced_data'] = reduced_data
                    st.success("PaCMAP completed!")

                st.session_state['page'] = "data_visualization"
            except Exception as e:
                st.error(f"Error performing dimensionality reduction: {e}")

    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")


# Strona do wizualizacji danych
def data_visualization_page():
    st.title("Data Visualization")
    if 'dataset_loaded' in st.session_state and st.session_state['dataset_loaded']:
        if st.button("Back to Technique Selection"):
            st.session_state['page'] = "choose_technique"

        if 'reduced_data' in st.session_state:
            fig, ax = plt.subplots()
            ax.scatter(st.session_state['reduced_data'][:, 0], st.session_state['reduced_data'][:, 1],
                       c=st.session_state['data']['target'], cmap='viridis')
            st.pyplot(fig)
        else:
            st.warning("No data to visualize. Please complete a reduction technique.")
    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")


def experiments_page():
    if 'dataset_loaded' in st.session_state and st.session_state['dataset_loaded']:
        st.title("Experiments")

        technique = st.selectbox("Select Reduction Technique for Experimentation", ["t-SNE", "UMAP", "TRIMAP", "PaCMAP"])

        if technique == "t-SNE":
            n_components_range = st.slider("Number of Components", 2, 3, (2, 3))
            perplexity_range = st.slider("Perplexity Range", 5, 50, (5, 50))
            learning_rate_range = st.slider("Learning Rate Range", 10, 200, (10, 200))

            param_combinations = list(product(
                range(n_components_range[0], n_components_range[1] + 1),
                range(perplexity_range[0], perplexity_range[1] + 1),
                range(learning_rate_range[0], 200, 10)
            ))

            progress_bar = st.progress(0)
            progress_steps = len(param_combinations)

            results = []
            for step, (n_components, perplexity, learning_rate) in enumerate(param_combinations):
                time.sleep(0.05)
                progress_bar.progress((step + 1) / progress_steps * 100)

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

            progress_bar = st.progress(0)
            progress_steps = len(param_combinations)

            results = []
            for step, (n_neighbors, min_dist) in enumerate(param_combinations):
                time.sleep(0.05)
                progress_bar.progress((step + 1) / progress_steps * 100)

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

            progress_bar = st.progress(0)
            progress_steps = len(param_combinations)

            results = []
            for step, n_neighbors in enumerate(param_combinations):
                time.sleep(0.05)
                progress_bar.progress((step + 1) / progress_steps * 100)

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

            progress_bar = st.progress(0)
            progress_steps = len(param_combinations)

            results = []
            for step, (n_components, n_neighbors) in enumerate(param_combinations):
                time.sleep(0.05)
                progress_bar.progress((step + 1) / progress_steps * 100)

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


def main():
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
        select_dataset_page()
    elif st.session_state.page == "Techniques Set Up":
        choose_technique_page()
    elif st.session_state.page == "View Data":
        data_visualization_page()
    elif st.session_state.page == "Experiments":
        experiments_page()


if __name__ == "__main__":
    main()
