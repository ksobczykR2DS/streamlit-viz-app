from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from trimap import TRIMAP
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from pacmap import PaCMAP


# Metryki
def compute_cf_nn(data_2d, labels, nn_max=100):
    nbrs = NearestNeighbors(n_neighbors=nn_max, algorithm='auto').fit(data_2d)
    distances, indices = nbrs.kneighbors(data_2d)

    cf_nn_values = []
    for i in range(len(data_2d)):
        label = labels[i]
        neighbors_labels = labels[indices[i]]
        cf_nn = np.sum(neighbors_labels == label) / nn_max
        cf_nn_values.append(cf_nn)

    return np.array(cf_nn_values)


def compute_cf(cf_nn_values):
    return np.mean(cf_nn_values)


# Eksperymenty
techniques_dict = {
    't-SNE': TSNE(),
    'UMAP': UMAP(),
    'TRIMAP': TRIMAP(),
    'PaCMAP': PaCMAP()
}

techniques_params = {
    't-SNE': {
        "perplexity": Integer(5, 100),
        "early_exaggeration": Integer(5, 25),
        "learning_rate": Integer(10, 1000),
        "n_iter": Integer(50, 1200),
        "metric": Categorical(["euclidean", "manhattan", "cosine"])
    },
    'UMAP': {
        "n_neighbors": Integer(10, 200),
        "min_dist": Real(0.0, 0.99),
        "metric": Categorical(["euclidean", "manhattan", "chebyshev", "minkowski", "canberra"])
    },
    'TRIMAP': {
        "n_inliers": Integer(2, 100),
        "n_outliers": Integer(1, 50),
        "n_random": Integer(1, 50),
        "weight_adj": Integer(100, 1000),
        "n_iters": Integer(50, 1200)
    },
    'PaCMAP': {
        "n_neighbors": Integer(10, 200),
        "mn_ratio": Real(0.1, 1.0),
        "fp_ratio": Real(1.0, 5.0)
    }
}


# W teorii ten kod powinien przyjmować dataset, sam wyciągać sobie label, a następnie zależy, jaka lista do niego
# przeprowadzać eksperymenty w ilości 'n_iter' przy maksymalizacji metryki 'cf' (ale to tylko teoria)
def perform_experiments(dataset, labels, technique_names_list, n_iter):
    print('Loading techniques...')
    model_dict = {}
    results = []

    x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.2,
                                                                                   random_state=42)
    try:
        for technique_name in technique_names_list.split(','):
            assert technique_name in techniques_dict.keys(), f'Unknown technique name: {technique_name}'
            model = techniques_dict[technique_name]
            model_dict[technique_name] = model

        for model_name, model in model_dict.items():
            print(f'Running experiments for {model_name}')
            pipe = Pipeline([('scaler', StandardScaler()), ('dimension_reduction', model)])

            def custom_scorer(pipe, X, y):
                transformed_data = pipe.fit_transform(X)
                cf_nn_values = compute_cf_nn(transformed_data, y, nn_max=100)
                cf_score = compute_cf(cf_nn_values)
                return cf_score

            scorer = make_scorer(custom_scorer, greater_is_better=True)

            opt = BayesSearchCV(
                estimator=pipe,
                search_spaces=techniques_params[model_name],
                n_iter=n_iter,
                random_state=7,
                scoring=scorer,
                verbose=True
            )

            opt.fit(x_train, y_train)

            best_score = opt.best_score_
            best_params = opt.best_params_
            results.append({'Model': model_name, 'Score': best_score, **best_params})

    except Exception as e:
        print(f'An error occurred: {e}')

    return results
