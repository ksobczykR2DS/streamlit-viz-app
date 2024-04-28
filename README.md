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
tab2 - ustawienie i wybór technik
tab3 - wizualizacje? 
tab4 - eksperymenty (maksymalizacja wyniku z wybranych metryk przy x iteracjach)



# Docker deployment
![docker_roadmap.png](documentation-resources%2Fdocker_roadmap.png)

Running docker file to build image:
    docker build -t my_streamlit_app .

Running app using docker:
    docker run -p 8502:8502 my_streamlit_app

To speed up the process:
* Docker Desktop allows you to allocate more CPU and memory resources to Docker via its settings
* Optimize your requirements.txt to avoid installing unnecessary or heavy packages