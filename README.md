# streamlit-viz-tool
Interactive app designed for advanced data visualization using techniques like t-SNE, UMAP, TRIMAP, and PaCMAP. It supports data loading, sampling, dynamic visualization, and quality metrics assessment.


Projekt:
Stworzenie przy pomocy biblioteki Streamlit narzędzia do wizualizacji danych. Powinno ono zawierać metody state-of-the-art tj. t-SNE, UMAP, TRIMAP, PaCMAP. Ma posiadać możliwość wczytywania danych, ich ograniczania (np. wzięcie tylko 10k sampli), wizualizacji wybraną metodą, liczenie wybranych (użytych na zajęciach)  dwóch metryk.


Implementacja: 
* Możliwość wyboru parametrów
* Okienko informacji co robi z każdy parametrów
* Domyślana wartość parametrów
* Ograniczenia parametrów do sensownych wartości

Ładowanie i samplowanie plików:
* Wbudowanie 3 datasety do testowania aplikacji (mnist, fminst etc.)
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

