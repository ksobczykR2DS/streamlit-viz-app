from datautils import *
import streamlit as st


st.set_page_config(page_title="Multi-Page App", page_icon=":memo:")


# Page functions
def load_page1():
    # można też z tego zrobić odrębna strone
    # Tutaj coś ładnie opisać o samej aplikacji i jak działa. Krótko i konkretnie.
    # Ew jakaś instrukcja obsługi programu i opis dataset'ów
    st.title("Dimensionality Reduction")
    st.write("""
            Interactive app designed for advanced data visualization using techniques like t-SNE, UMAP,
            TRIMAP, and PaCMAP. It supports data loading, sampling, dynamic visualization, and quality metrics assessment.
        """)

    #st.write("### MNIST Handwritten Digits")
    #st.write("Dataset of 70,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.")

    #st.write("### 20 Newsgroups Text Data")
    #st.write("Collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.")

    #st.write("### Labeled Faces in the Wild (LFW)")
    #st.write(
    #    "More than 13,000 images of faces collected from the web, each labeled with the name of the person pictured.")

    # To wyżej zdecydowanie do poprawy
    # + możliwość samplowania
    dataset_names = [
        'MNIST Handwritten Digits',
        '20 Newsgroups Text Data',
        'Labeled Faces in the Wild (LFW)'
    ]

    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names)

    if st.button("Load chosen dataset"):
        if selected_dataset == 'MNIST Handwritten Digits':
            dataset = load_mnist_dataset()
        elif selected_dataset == '20 Newsgroups Text Data':
            dataset = load_20_newsgroups_dataset()
        elif selected_dataset == 'Labeled Faces in the Wild (LFW)':
            dataset = load_lfw_dataset()

        if dataset is not None:
            st.session_state['data'] = dataset
            st.session_state['dataset_loaded'] = True
            st.success("Dataset loaded successfully. Navigate to 'Choose Technique and Parameters' to continue.")


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
