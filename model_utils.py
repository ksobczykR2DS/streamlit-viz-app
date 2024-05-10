import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np


# trzeba zredukowac wymiary do 3 i potem użyć plotly

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def preprocess_data(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    return data


def get_embeddings(model, data):
    try:
        data = preprocess_data(data)
        _ = model(np.zeros((1, data.shape[1])))
        intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                                  outputs=model.get_layer('embedding_layer').output)
        embeddings = intermediate_layer_model.predict(data)
        return embeddings
    except Exception as e:
        st.error(f"Error in generating embeddings: {e}")
        return None


def reduce_dimensions(embeddings, components=3):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=components)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)
    return reduced_embeddings


def visualize_embeddings(embeddings, labels):
    df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
    df['label'] = labels
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')
    st.plotly_chart(fig, use_container_width=True)
