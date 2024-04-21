import streamlit as st
from datautils import *

# To run up as website (with venv activated)
# USAGE:  streamlit run app.py
if 'dataset_loaded' not in st.session_state:
    st.session_state['dataset_loaded'] = False
    st.session_state['data'] = None


def unlock_tabs():
    st.session_state['tab1_submitted'] = True


tab1, tab2, tab3 = st.tabs(["Select Dataset", "Choose Technique and Parameters", "Data Visualization"])

with tab1:
    st.header("Dataset Selection")
    dataset_name = st.selectbox("Choose a dataset or upload your own",
                                ["MNIST", "Fashion-MNIST", "CIFAR-10", "Upload Dataset"])

    if dataset_name == "Upload Dataset":
        uploaded_file = st.file_uploader("Choose a file (CSV or Excel)")
        if uploaded_file:
            data = upload_file(uploaded_file)
            if isinstance(data, str):
                st.error(data)
            else:
                st.session_state['data'] = data
                st.session_state['dataset_loaded'] = True
    else:
        if st.button("Load Dataset"):
            data = load_dataset(dataset_name)
            if data is not None:
                st.session_state['data'] = data
                st.session_state['dataset_loaded'] = True

with tab2:
    if st.session_state['dataset_loaded']:
        st.header("Choose Technique and Parameters")
        st.write("Dataset successfully loaded!")
    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")

with tab3:
    if st.session_state['dataset_loaded']:
        st.header("Data Visualization")
        st.write("Analysis and visualization tools could be added here.")
    else:
        st.error("Please load a dataset in the 'Select Dataset' tab first.")

