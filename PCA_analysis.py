import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go


def perform_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    return components, pca.explained_variance_ratio_


def plot_pca(components, labels=None):
    fig = px.scatter(x=components[:, 0], y=components[:, 1], color=labels, labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'})
    fig.update_layout(title="PCA Components Analysis", xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, use_container_width=True)

def plot_pca_3d(components, labels=None):
    df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    if labels is not None:
        df['Label'] = labels
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Label', title="3D PCA Visualization")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

def plot_pca_biplot(features, labels=None):
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)  # Przetwarzanie wewnątrz funkcji
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(x=components[:, 0], y=components[:, 1], color=labels,
                     labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'})
    for i, feature in enumerate(features.columns):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    fig.update_layout(
        title="PCA Biplot",
        xaxis_title="PC 1",
        yaxis_title="PC 2"
    )
    st.plotly_chart(fig, use_container_width=True)



def plot_explained_variance(explained_variance_ratio):
    cum_var = np.cumsum(explained_variance_ratio)
    fig = go.Figure(data=[
        go.Bar(name='Individual', x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))], y=explained_variance_ratio),
        go.Scatter(name='Cumulative', x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))], y=cum_var)
    ])
    fig.update_layout(title="Explained Variance by PCA Components",
                      xaxis_title="Components",
                      yaxis_title="Explained Variance")
    st.plotly_chart(fig, use_container_width=True)


def plot_pca_loadings_heatmap(pca, feature_names):
    loadings = pca.components_
    fig = px.imshow(loadings, x=feature_names, y=[f"Component {i+1}" for i in range(loadings.shape[0])],
                     labels=dict(x="Feature", y="Component", color="Loading"))
    fig.update_layout(title="PCA Component Loadings")
    st.plotly_chart(fig, use_container_width=True)

