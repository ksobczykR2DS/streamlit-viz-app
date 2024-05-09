# TODO, Dokumentacja

**Wszystko po angielsku**

# Opis projektu, zawartość
Interactive app designed for advanced data visualization using state-of-art techniques like t-SNE, UMAP, TRIMAP, and PaCMAP. 
It supports data loading, sampling, dynamic visualization, and quality metrics assessment.

# Sposób odpalenia (throught python and docker)
---
# Charakterystyka technik i ich parametrów
---
# Omówienie każdej ze stron + skriny
---
# Opis działania użytej metryki
# Opis możliwości interpretacji wyniku


Projekt:
Stworzenie przy pomocy biblioteki Streamlit narzędzia do wizualizacji danych. Powinno ono zawierać metody state-of-the-art tj. t-SNE, UMAP, TRIMAP, PaCMAP. Ma posiadać możliwość wczytywania danych, ich ograniczania (np. wzięcie tylko 10k sampli), wizualizacji wybraną metodą, liczenie wybranych (użytych na zajęciach)  dwóch metryk.

Implementacja: 
* Możliwość wyboru parametrów
* Okienko informacji co robi z każdy parametrów
* Domyślana wartość parametrów
* Ograniczenia parametrów do sensownych wartości

Ładowanie i samplowanie plików:
* Wbudowanie 3 datasety do testowania aplikacji (mnist, fminst etc.) 
(wykorzystanie ich w celach nauki i potestowania technik, albo wgranie swojego i faktycznie użycie technik)
* Ograniczenie samplowania do len(dataset)
* Możliwość ustawienia sample na 20%/40% etc.
* Ładowanie własnych datasetów

Analiza:
* możliwość przeprowadzania kilku analiz na raz
* możliwość porównywania wyników na jednym oknie
* porównywanie modeli i eksperymentów za pomocą metryk
* możliwość eksperymentów z wieloma wersjami parametrów (gridsearch, randomsearch, baysian search o ile sie da)


t-SNE
UMAP
TRIMAP
PaCMAP

Schematy UI:
3-4 zakładki - pierwsza do wyboru lub wgrania datasetu, bez wgrania datasetu reszta zakładake zablokowana

tab1 - datasety
tab2 - ustawienie i wybór technik i wizualiacja
tab3 - eksperymenty (maksymalizacja wyniku z wybranych metryk przy x iteracjach)



# Docker deployment
![docker_roadmap.png](documentation-resources%2Fdocker_roadmap.png)

Running docker file to build image:
    docker build -t my_streamlit_app .

Running app using docker:
    docker run -p 8502:8502 my_streamlit_app

To speed up the process:
* Docker Desktop allows you to allocate more CPU and memory resources to Docker via its settings
* Optimize your requirements.txt to avoid installing unnecessary or heavy packages




# Techniques
t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE to technika redukcji wymiarowości, która jest szczególnie dobra w wizualizacji danych o wysokiej wymiarowości na niskowymiarowe przestrzenie (zazwyczaj 2D lub 3D). Jest często używana do wizualizacji w dziedzinach takich jak bioinformatyka i uczenie maszynowe. Metoda ta działa poprzez próbę zachowania podobieństwa lokalnych sąsiedztw w przestrzeni wielowymiarowej, przekładając je na przestrzeń o niższej wymiarowości. t-SNE skupia się bardziej na lokalnych strukturach danych, co czasami prowadzi do ignorowania niektórych globalnych wzorców w danych.

UMAP (Uniform Manifold Approximation and Projection)
UMAP to stosunkowo nowa technika redukcji wymiarowości, która może być używana zarówno do redukcji wymiarów, jak i do celów uczenia nienadzorowanego. Jest podobna do t-SNE w tym sensie, że obie techniki starają się zachować lokalne struktury danych. Jednak UMAP opiera się na matematycznej koncepcji Równomiernego Rozkładu i Projektowania Różniczkowalnych Mnogości, co pozwala jej lepiej zachować zarówno lokalne, jak i globalne struktury danych. Jest również zazwyczaj szybsza w stosunku do t-SNE i może lepiej radzić sobie z większymi zbiorami danych.

TriMAP
TriMAP to technika redukcji wymiarów oparta na tripletach punktów, podobna do t-SNE. TriMAP skupia się na minimalizacji straty, która zachowuje relacje oparte na tripletach: punkt odniesienia jest bliższy punktom wewnątrzklasowym niż punktom międzyklasowym. Jest to szczególnie użyteczne w przypadkach, gdy dane mają jasno zdefiniowane grupy lub klasy, i może skutecznie utrzymywać lokalne odległości między punktami danych.

PaCMAP (Pairwise Controlled Manifold Approximation and Projection)
PaCMAP jest nowoczesnym podejściem do redukcji wymiarowości, które, podobnie jak UMAP i t-SNE, koncentruje się na zachowaniu lokalnych struktur danych. Jednak wprowadza nową ideę kontroli par punktów w celu zachowania zarówno bliskich, jak i dalekich relacji, dzięki czemu utrzymuje równowagę między lokalnymi a globalnymi strukturami danych. Ta metoda została zaprojektowana, aby być odporna na typowe problemy, takie jak zbieganie do nieprzydatnych, jednopunktowych rozwiązań, które mogą wystąpić w innych technikach.

---
# Parameters
**t-SNE (t-Distributed Stochastic Neighbor Embedding)**

n_components: - default 2 (cannot be changed)
* The dimension of the embedded space.
* Default is usually 2 for visualization.

perplexity: - default 30, range 5-100
* The number of nearest neighbors that t-SNE considers.
* Typical values between 5 and 50.

early_exaggeration: - default 12, range 5, 25
* Controls how tight natural clusters in the original space are in the embedded space.
* Larger values increase the space between clusters.

learning_rate: - default 'auto', range 10:1000
* The learning rate for optimization.
* Typical values between 10 and 1000.

n_iter: default 300 - range 50, 1200
    Maximum number of iterations for the optimization.
    Usually at least 250.

* metric:
The distance metric to use. Common options include "euclidean", "manhattan", and "cosine".

---

**UMAP (Uniform Manifold Approximation and Projection)**
n_neighbors: - default 15, range 10-200
*  size of the local neighborhood used for manifold approximation.
* Typical values between 10 and 200.

n_components: - 2 cannot be changed
* The dimension of the space to embed into.
* Default is 2 for visualization.

min_dist:
* The minimum distance apart that points are allowed to be in the low-dimensional representation.
* Smaller values will result in a more clustered/clumped embedding.

metric:
* The metric to use to compute distances in high dimensional space.
* Can be "euclidean", "manhattan", "chebyshev", "minkowski", "canberra", and many more.

learning_rate:
* The learning rate for optimization.

---

**TriMAP**
n_components:
* The dimension of the space to embed into.
* Typically 2 for visualization.

n_inliers:
* The number of inlier points for triplet constraints.
* Default is 10.

n_outliers:
* The number of outlier points for triplet constraints.
* Default is 5.

n_random:
* The number of random triplet constraints per point.
* Default is 5.

weight_adj:
* The weighting factor that adjusts the importance of the triplet constraints.
* Default is 500.

n_iters:
* Number of iterations for optimization.

---

**PaCMAP (Pairwise Controlled Manifold Approximation and Projection)**
n_components:
* The number of dimensions to project down to.
* Typically 2 or 3 for visualization purposes.

n_neighbors:
* The number of nearest neighbors for manifold learning.
* Similar to UMAP and t-SNE's approach.

mn_ratio:
* The ratio of mid-near pairs to mid-far pairs.
* Default is 0.5.

fp_ratio:
* The ratio of far-positive pairs in the objective function.
* Default is 2.0.

# Components Analysis

# Docker deployment
![docker_roadmap.png](documentation-resources%2Fdocker_roadmap.png)

Running docker file to build image:
    docker build -t my_streamlit_app .

Running app using docker:
    docker run -p 8502:8502 my_streamlit_app

To speed up the process:
* Docker Desktop allows you to allocate more CPU and memory resources to Docker via its settings
* Optimize your requirements.txt to avoid installing unnecessary or heavy packages