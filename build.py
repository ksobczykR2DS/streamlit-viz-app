import matplotlib.pyplot as plt
import streamlit as st
from datautils import *
from itertools import product
import time


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

    st.write("### MNIST Handwritten Digits")
    st.write("Dataset of 70,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.")

    st.write("### 20 Newsgroups Text Data")
    st.write("Collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.")

    st.write("### Human Activity Recognition")
    st.write(
        "Recordings of 30 study participants performing activities of daily living with a waist-mounted smartphone with embedded inertial sensors.")

    st.write("### Gene Expression Cancer RNA-Seq")
    st.write(
        "A random extraction of gene expressions of patients having different types of tumor: BRCA, KIRC, COAD, LUAD, and PRAD.")

    st.write("### Curated Breast Imaging Subset")
    st.write("Features for breast cancer histopathology images.")

    st.write("### Labeled Faces in the Wild (LFW)")
    st.write(
        "More than 13,000 images of faces collected from the web, each labeled with the name of the person pictured.")

    # To wyżej zdecydowanie do poprawy
    # + możliwość samplowania
    dataset_names = [
        'MNIST Handwritten Digits',
        '20 Newsgroups Text Data',
        'Human Activity Recognition',
        'Gene Expression Cancer RNA-Seq',
        'Curated Breast Imaging Subset',
        'Labeled Faces in the Wild (LFW)'
    ]

    selected_dataset = st.selectbox("Choose a dataset to load", dataset_names)

    if st.button("Load chosen dataset"):
        if selected_dataset == 'MNIST Handwritten Digits':
            load_mnist_dataset()
        elif selected_dataset == '20 Newsgroups Text Data':
            load_20_newsgroups_dataset()
        elif selected_dataset == 'Human Activity Recognition':
            load_human_activity_dataset()
        elif selected_dataset == 'Gene Expression Cancer RNA-Seq':
            load_gene_expression_dataset()
        elif selected_dataset == 'Curated Breast Imaging Subset':
            load_breast_imaging_dataset()
        elif selected_dataset == 'Labeled Faces in the Wild (LFW)':
            load_lfw_dataset()


def load_page2():
    st.title("Page 2")
    st.write("This is the content of page 2.")


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