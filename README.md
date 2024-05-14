# Project Description and Contents

This project presents an interactive application tailored for advanced data visualization utilizing **state-of-the-art** techniques such as t-SNE, UMAP, TriMAP, and PaCMAP. 
Designed to cater to both beginners and experts in data science, the app offers robust functionality including data loading, dynamic visualization through real-time adjustments, data sampling, and the evaluation of various quality metrics. 
These features empower users to explore complex datasets comprehensively, providing insightful visual representations that facilitate better understanding and analysis of the data's underlying structure and relationships.

Authors: Kacper Sobczyk, Katarzyna Główczyk

---

# Application Set Up

To run the application, you need to set up a Python environment, either locally or in a Docker container as specified in the Docker section.

## Python Environment Setup 
First, ensure you have a Python virtual environment set up and all dependencies installed. Once the virtual environment is active and all dependencies from requirements.txt are installed, you can run the application on your local machine using:
 ```
streamlit run app.py --server.port 8502
 ```
Your application should be accessible at http://localhost:8502

## Docker

Begin by building a Docker image from your Dockerfile. Open your terminal, navigate to the directory containing your Dockerfile, and run the following command (replace app with the name you want to give your Docker image):
 ```
docker build -t app .
 ```
After the image is built, you can run it as a container. To do so, use the following command:
 ```
docker run -p 8502:8502 app
 ```
Once the container is running, open a web browser and visit http://localhost:8502. You should see your Streamlit application running.

---


# Project Overview

1. **Load Dataset**

![img.png](resources/img.png)


The initial page of our web application focuses on dataset loading and visualization, offering options to either choose from predefined datasets (e.g., MNIST, Fashion-MNIST, Scene Dataset) or upload custom datasets in CSV, XLSX, or XLS formats.
Custom 

Dataset Uploading

A user-friendly upload interface is activated when the "Upload Dataset" option is selected. This allows users to bring their specific data into the application for analysis.

Data Sampling

The application features a sampling slider, enabling users to specify the percentage of the dataset to load (ranging from 1% to 100%). This functionality is essential for handling large datasets efficiently by adjusting the sample size according to system capabilities or analytical needs.

Data Preview

After dataset selection and loading, the application displays a preview of the first few rows. Users can further explore the data through a feature distribution visualization option, which provides insights into the selected data column.

![img_1.png](resources/img_1.png)

2. **Techniques Set Up and Visualization**

The second page of our web application focuses on the selection of dimensionality reduction techniques and the customization of their respective parameters. This allows users to tailor the data analysis process according to their specific needs and dataset characteristics.

![img_3.png](resources/img_3.png)

Visual Feedback on Dataset Loading

At the beginning of the page, users receive visual confirmation of the dataset currently loaded, displayed prominently in a styled box. This feedback ensures that users are aware of the data context as they proceed with technique selection and parameter tuning.

Selection of Dimensionality Reduction Techniques

The page provides an expandable interface where users can select from multiple dimensionality reduction techniques: t-SNE, UMAP, TRIMAP, and PaCMAP. Each technique section offers a brief description and the option to activate the technique for the current session.

![img_2.png](resources/img_2.png)

3. **PCA Components Analysis**

The third page of our application is dedicated to the Principal Component Analysis (PCA), enabling users to perform and visualize PCA on datasets they have loaded and configured. This page is structured to guide users through a systematic analysis, offering options for detailed component exploration and data visualization.

Feature Selection for PCA

Users can choose to perform PCA using all features of the dataset or select specific features. A checkbox activates a multi-select dropdown menu where specific features can be chosen. This flexibility allows for focused analysis on relevant features, potentially improving the clarity and relevance of the PCA results.
PCA Configuration

A slider allows users to choose the number of principal components they wish to analyze, with the maximum limit set by the smaller of the number of features or the number of data points to ensure a valid PCA computation.

PCA Execution and Visualization Options

Three checkboxes provide users with options to visualize different aspects of PCA:

* PCA Plot: Depending on the number of components selected (two or three), a 2D or 3D PCA plot is generated, providing a visual representation of data spread in the principal component space. 

![img_4.png](resources/img_4.png)

* Explained Variance Plot: This plot shows the proportion of the dataset’s variance that each principal component accounts for, helping users understand the importance of each component. 

![img_5.png](resources/img_5.png)

* Loadings Heatmap: A heatmap of PCA loadings can be generated, which is useful for identifying which variables contribute most to each principal component.

![img_7.png](resources/img_7.png)

4. **Technique Tuning with RandomSearch**


## Result Interpretation

### Cluster Formation
- **Observation**: Clusters of data points in a 2D or 3D PCA scatter plot.
- **Interpretation**: Clusters indicate that PCA has successfully reduced the dimensionality of the data while preserving the variance. Each cluster may represent a different group, category, or class within the data. This can be useful for identifying natural groupings and patterns.

### Outliers Detection
- **Observation**: Data points that are distant from the main clusters in the scatter plot.
- **Interpretation**: These points can be considered outliers, indicating observations that significantly differ from the majority of the data. Identifying outliers is crucial for understanding the dataset's variability and can inform data cleaning and preprocessing steps.

### Class Overlap
- **Observation**: Overlapping clusters in the PCA plot.
- **Interpretation**: Overlap between clusters suggests that some classes or groups are not completely separable in the reduced dimensional space. This might imply that there are similarities between these groups or that the features used do not completely differentiate between them.

### Explained Variance
- **Observation**: Bar plot showing the explained variance of each principal component, often accompanied by a cumulative variance line plot.
- **Interpretation**: The explained variance plot shows how much variance each principal component captures. A steep rise at the beginning indicates that the first few components capture most of the variance, making them the most informative. The cumulative variance plot helps determine how many components are needed to capture a desired amount of total variance, indicating the effectiveness of the dimensionality reduction.

### Component Loadings
- **Observation**: Heatmap or bar plot showing the loadings (contributions) of each original feature to the principal components.
- **Interpretation**: Features with high absolute loading values are more influential in forming the principal components. This helps identify the most important variables in the dataset. For example, a heatmap can reveal which features contribute the most to each component, aiding in feature selection and understanding the underlying structure.

### Statistical Description
- **Observation**: Summary statistics (mean, standard deviation, min, max, etc.) of the PCA components.
- **Interpretation**: Descriptive statistics provide insights into the distribution of the data in the reduced space. This helps understand the central tendency and variability of the transformed data, which can be useful for further analysis or model building.

### Correlation Analysis
- **Observation**: Correlation matrix of the PCA components.
- **Interpretation**: The correlation matrix helps understand the linear relationships between the components. High correlation values between components might indicate redundancy, whereas low correlation values suggest that the components capture different aspects of the data's variability. This is useful for evaluating the effectiveness of the PCA in reducing dimensionality without losing important information.

---
# DETAILS

## Characteristics of Used Techniques and Their Parameters

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) is a technique that converts similarities between data points into joint probabilities and then minimizes the Kullback-Leibler divergence between these probabilities in both the high-dimensional and low-dimensional spaces. This makes it exceptionally good at creating a two- or three-dimensional map of datasets, revealing intrinsic patterns such as clusters. t-SNE is particularly sensitive to local structure and has a tendency to create distinct groups or clusters, making it useful for visualizing clusters of high-dimensional data.

**UMAP** (Uniform Manifold Approximation and Projection) operates under the assumption that the data is uniformly distributed on a locally connected Riemannian manifold and uses a fuzzy topological structure to project the data into a low-dimensional space. UMAP maintains both local and more global structure of the data better than t-SNE, often providing superior visualization outcomes for more complex datasets. It is generally faster than t-SNE and scales more favorably with the size of the input data.

**TriMAP** is another method that focuses on preserving the global relationships between data points alongside local ones, using triplet-based loss functions to guide the embedding. It selects triplets of points—consisting of an anchor, a near point, and a far point—and optimizes the layout to keep the relative distance consistent between the high-dimensional and the reduced space. This approach helps in maintaining the true geodesic distances among points, which is especially useful for datasets where global relationships are critical.

**PaCMAP** (Pairwise Controlled Manifold Approximation and Projection) is a relatively new method that also seeks to maintain a balance between local and global structures in the data. It uses pairwise constraints and controls the ratio of certain types of point pairs to steer the manifold learning process. This approach helps in reducing the tendency of clustering effect seen in methods like t-SNE, leading to a more uniform distribution of points in the embedded space, which can be more informative for certain types of analytical tasks.

## Paramethers
**t-SNE (t-Distributed Stochastic Neighbor Embedding) Parameters**
* n_components: Specifies the dimension of the output space; commonly set to 2 for visualization. 
* perplexity: Influences the number of nearest neighbors in the high-dimensional space, affecting local versus global data structure balance. 
* early_exaggeration: Amplifies differences in densities, helping to separate clusters more distinctly in the embedding. 
* learning_rate: Affects the step size at each iteration while minimizing the cost function, crucial for convergence of the algorithm. 
* n_iter: Sets the maximum number of optimization iterations, determining how long the algorithm runs. 
* metric: Determines the method used to measure distances between points in high-dimensional space, impacting the shape of the embedding.

**UMAP (Uniform Manifold Approximation and Projection) Parameters**
* n_neighbors: Controls the size of the local neighborhood considered for manifold approximation, crucial for capturing the data’s underlying structure. 
* n_components: Sets the dimensionality of the target embedding space, usually 2 for easy visualization. 
* min_dist: Regulates the tightness of the embedding, influencing how closely points can cluster together. 
* metric: Defines the distance metric for calculating dissimilarities between data points, which can dramatically affect the embedding's structure. 
* learning_rate: Adjusts the learning rate during optimization, important for effectively minimizing the embedding’s loss function.

**TriMAP Parameters**
* n_components: Indicates the dimensionality of the target projection space, with 2 being common for visual purposes. 
* n_inliers: Specifies the number of close points considered when forming triplet constraints, affecting local data structure preservation. 
* n_outliers: Determines the number of distant points used in triplet constraints, helping to maintain global data relationships. 
* n_random: Sets the number of random triplets per point, which can enhance the robustness of the manifold learning. 
* weight_adj: Adjusts the weight given to triplet constraints during optimization, balancing the emphasis on local versus global features. 
* n_iters: Dictates the number of iterations for optimizing the embedding, with more iterations potentially leading to better embeddings.

**PaCMAP (Pairwise Controlled Manifold Approximation and Projection) Parameters**
* n_components: Defines the number of dimensions in the reduced space, often set to 2 or 3 for visualization. 
* n_neighbors: Influences the manifold learning by specifying how many nearest neighbors to consider, similar to parameters in UMAP and t-SNE. 
* mn_ratio: Controls the ratio of medium-near to medium-far point pairs, adjusting the local versus global emphasis in the embedding. 
* fp_ratio: Sets the balance of far-positive pairs in the objective function, crucial for defining how distant relationships affect the embedding's structure.


##  Optimization of Technique Parameters (Random Search)

## Description of the Used Metric

![img_6.png](resources/img_6.png)

### Euclidean Distance
The Euclidean distance, also known as the L2 distance, is used to determine the natural distance between two points in a Euclidean space. It is calculated as the square root of the sum of the squares of the differences between corresponding coordinates of the points. For two points A and B in an n-dimensional space, the formula is: d(A, B) = sqrt((A1 - B1)^2 + (A2 - B2)^2 + ... + (An - Bn)^2).

### Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, disregarding their magnitude. This measure is used in applications such as text analysis, focusing on the direction rather than the length of the vectors. The formula to calculate the similarity between two vectors A and B is: similarity(A, B) = (A . B) / (||A|| * ||B||), where A . B is the dot product of the vectors, and ||A|| and ||B|| are the norms of these vectors.

### Manhattan Distance
The Manhattan distance, also known as the L1 distance or city block distance, measures the total sum of the absolute differences between the coordinates of two points. It is often used in grid-like path calculations, such as in city road networks. For two points A and B in an n-dimensional space, the Manhattan distance is calculated as: d(A, B) = |A1 - B1| + |A2 - B2| + ... + |An - Bn|.

### Chebyshev Distance
The Chebyshev distance, also known as the chessboard distance or infinity norm (L∞), measures the maximum absolute difference between the coordinates of two points in multi-dimensional space. It is particularly useful in applications where movements along grid axes and diagonals are allowed, such as in a chess game where the king can move exactly one square in any direction (vertical, horizontal, or diagonal). For two points A and B in an n-dimensional space, the Chebyshev distance is defined as the maximum of the absolute differences of their corresponding coordinates. In simpler terms, it can be expressed as:

Chebyshev distance between points A and B = max(|A1 - B1|, |A2 - B2|, ..., |An - Bn|)



