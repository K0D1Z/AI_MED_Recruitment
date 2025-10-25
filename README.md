# Rozwiązanie zadania rekrutacyjnego do KN AI Med
Powyższe zadanie rekrutacyjne było moim pierwszym zetknięciem z Machine Learningiem w praktyce. Niezależnie od wyniku rekrutacji dziękuję za możliwość poszerzenia swojej wiedzy w tej dziedzinie. 

## Moje podejście do problemu
Postanowiłem w praktyce porównać ze sobą wszystkie pięć modeli zaproponowanych w pliku ML.md, aby znaleźć te, które dla danych z pliku task_data.csv dają najlepsze rezultaty. Takie podejście zaowocowało znalezieniem dwóch z pięciu modeli, które wyraźnie lepiej radzą sobie z wykrywaniem kardiomegalii na podstawie zestawu danych z dostarczonego pliku.

Lista sprawdzanych modeli:
- Decision Tree - decyzja o klasyfikacji oparta o serię pytań w strukturze przypominającej drzewo, węzły drzewa reprezentują testy danych cech, a liście reprezentują wyniki tych testów; tutaj skalowanie cech nie jest zazwyczaj konieczne.
- Random Forest - algorytm budujący wiele drzew decyzyjnych opartych o podzbiory danych; tutaj skalowanie cech nie jest zazwyczaj konieczne.
- K-Nearest Neighbors - algorytm patrzy na k najbliższych sąsiadów z danych treningowych, następnie sprawdza, do których klas należą dani sąsiedzi i na tej podstawie klasyfikuje próbkę; niezbędne przeskalowanie cech.
- Support Vector Machine (SVM) - algorytm znajduje linię (hiperpłaszczyznę), która separuje klasy, a następnie maksymalizuje margines między klasami; dane są klasyfikowane w zależności, po której stronie marginesu się znajdą; niezbędne przeskalowanie cech.
- Logistic Regression - algorytm również znajduje hiperpłaszczyznę, a następnie oblicza odległość punktu od marginesu i szacuje prawdopodobieństwo przynależności próbki do danej klasy; niezbędne przeskalowanie cech.

## Moje rozwiązanie problemu
Przeprowadziłem wstępne przetwarzanie danych oraz ich standaryzację poprzez konwersję na odpowiednie typy oraz przeskalowanie cech poprzez użycie klasy StandardScaler. 

Następnie przeprowadziłem walidację krzyżową danych na pięciu testach z użyciem StratifiedKFold. Użyłem tego rozwiązania, aby zapewnić stałe proporcje etykiet (0 - dla osób zdrowych, 1 - dla osób chorych). Dla małego zbioru danych, który został dostarczony, standardowy KFold mógłby nie zadziałać, ponieważ mogłoby dojść do sytuacji, w której do testów zostanie losowo wybranych nieproporcjonalnie dużo danych o chorych lub zdrowych osobach, przez co test stałby się nierelewantny w stosunku do wszystkich danych.

Po przyuczeniu modelu i dokonaniu przez model predykcji przeszedłem do testów. Dla mojego rozwiązania wybrałem proste testy:
- Accuracy Score - standardowy test sprawdzający stosunek prawidłowych predykcji do ogólnej liczby testów.
- Precision Score - sprawdzający stosunek prawidłowych diagnoz do liczby wszystkich przewidzianych pozytywnie wyników (false positives).
- Recall Score - sprawdzający stosunek prawidłowych diagnoz do wszystkich wystąpień pozytywnych wyników (false negatives).
- F1 Score - średnia harmoniczna między Precision Score i Recall Score.

Z wyników pięciu przejść pętli w etapie walidacji krzyżowej wyciągnąłem średnią arytmetyczną każdego testu dla pięciu modeli.

Na koniec zestawiłem z sobą w postaci tabeli wyniki czterech wybranych testów oraz wybrałem dwa najlepiej działające modele.

## Wyniki
Z zestawienia wyników wynika, że najlepiej radzącymi sobie modelami dla dostarczonego zestawu danych są KNeighbors Classifier oraz Support Vector Machine (SVC).

| Model Name | Accuracy Score | Precision Score | Recall Score | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| KNeighbors Classifier | 0.789286 | 0.814286 | 0.933333 | 0.866434 |
| SVC | 0.785714 | 0.823810 | 0.933333 | 0.868765 |
| Random Forest Classifier | 0.732143 | 0.814286 | 0.866667 | 0.826434 |
| Logistic Regression | 0.650000 | 0.802857 | 0.760000 | 0.760000 |
| Decision Tree Classifier | 0.600000 | 0.760000 | 0.693333 | 0.715152 |

## Możliwości dalszego rozwoju
Po znalezieniu dwóch najlepszych modeli powinienem zdobyć więcej danych testowych, ponieważ 40 wierszy to zdecydowanie za mało, aby stworzyć model, który z dużą skutecznością będzie mógł wykrywać kardiomegalię. Dodatkowo mogę zastosować strojenie hiperparametrów dla modeli KNeighbors Classifier oraz SVC.

## Kilka słów ode mnie
Jestem studentem pierwszego roku na kierunku Informatyka i Systemy Inteligentne na wydziale EAIIiB. Od początku nauki w technikum interesowałem się programowaniem w Pythonie oraz programowaniem webowym, lecz nie miałem szansy pracować nad dużymi projektami. 

W Waszym kole widzę szansę na zdobycie tego kluczowego doświadczenia i rozwój umiejętności. Program moich studiów jest zbieżny z potencjalnymi zadaniami, dlatego mój wkład w projekty będzie systematycznie rósł, napędzany zarówno postępami na kolejnych semestrach studiów, jak i nauką oraz pracą w kole.

Jestem zdeterminowany, by szybko uzupełniać wszelką niezbędną wiedzę i aktywnie angażować się w realizowane zadania.