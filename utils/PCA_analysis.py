import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import io
import plotly.express as px


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
