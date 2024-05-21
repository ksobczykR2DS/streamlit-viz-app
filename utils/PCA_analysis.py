import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import io
import plotly.express as px
from sklearn.decomposition import KernelPCA


def perform_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    return components, pca.explained_variance_ratio_


def perform_kernel_pca(data, n_components, kernel):
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    components = kpca.fit_transform(data)
    variance_ratio = [1] * n_components
    return components, variance_ratio


def plot_pca(components, labels=None, data_index=None):
    df = pd.DataFrame(components, columns=['Principal Component 1', 'Principal Component 2'])
    if labels is not None:
        df['Label'] = labels
    if data_index is not None:
        df['ID'] = data_index

    hover_data = ['ID'] if 'ID' in df.columns else None
    fig = px.scatter(df, x='Principal Component 1', y='Principal Component 2', color='Label',
                     hover_data=hover_data, title="PCA Components Analysis")
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(clickmode='event+select')
    st.plotly_chart(fig, use_container_width=True)


def plot_pca_3d(components, labels=None, data_index=None):
    df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    if labels is not None:
        df['Label'] = labels
    if data_index is not None:
        df['ID'] = data_index

    hover_data = ['ID'] if 'ID' in df.columns else None
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Label',
                        hover_data=hover_data, title="3D PCA Visualization")
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
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
    loadings_df = pd.DataFrame(loadings, columns=feature_names,
                               index=[f"Component {i + 1}" for i in range(loadings.shape[0])])

    fig = px.imshow(loadings_df, text_auto=True,
                    labels=dict(x="Feature", y="Component", color="Loading"),
                    x=feature_names,
                    y=[f"Component {i + 1}" for i in range(loadings.shape[0])])
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title="PCA Component Loadings",
        xaxis_title="Features",
        yaxis_title="Components",
        coloraxis_colorbar=dict(
            title="Loading"
        ),
        autosize=False,
        width=1000,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def calculate_statistics(components_df):
    description = components_df.describe()
    return description


def perform_advanced_analysis(components_df):
    correlation = components_df.corr()
    return correlation


def export_analysis(components_df):
    stats = calculate_statistics(components_df)
    correlation = perform_advanced_analysis(components_df)

    output = io.StringIO()
    stats.to_csv(output, mode='a')
    output.write("\nCorrelation Matrix:\n")
    correlation.to_csv(output, mode='a')

    output.seek(0)
    return output.getvalue()
