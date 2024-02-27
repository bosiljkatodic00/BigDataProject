# BigDataProject

##Sažetak:

U ovom projektu sam pokušala predvidjeti kašnjenja letova u SAD-u.
Iako bi ovo takođe moglo biti problem regresije koji bi predviđao dužinu kašnjenja u minutama, posmatrala sam ga kao binarni klasifikacioni problem.
Ciljna promjenjiva je promjenljiva "delayed" koja za svako kašnjenje veće od  14 minuta ima vrijednost 1.

Pokrenula sam nekoliko različitih modela klasifikacije, a najtačniji su se pokazali Random Forest Classifier i XGBoost Classifier, sa AUC ocjenom 0.59. 

-Skupovi podataka:

Koristila sam dva izvora podataka sa sajta Kaggle:
1. Svi letovi u SAD-u od avgusta 2013. godine do avgusta 2023. godine

2. Svi letovi u SAD-u u toku 2015. godine

-Vizuelizacije i izvještaji:

Prvi skup podataka sam koristila da generišem grafikone koji bi korisnicima dali uvid u performanse aerodroma i avio-kompanija.
Kreirani su:

1. Interaktivni dijagrami:
Grafikon broja letova po avio-kompaniji: koristi se Pie grafikon koji prikazuje raspodjelu broja letova po avio-kompanijama.
![newplot (2)](https://github.com/bosiljkatodic00/BigDataProject/assets/151973670/6b937223-f693-46f1-bf15-8e232eab190d)

Grafikon procentualnog udjela otkazanih letova po avio-kompaniji: takođe koristi Pie grafikon za prikaz procentualnog udjela otkazanih letova po avio-kompanijama.
![newplot (3)](https://github.com/bosiljkatodic00/BigDataProject/assets/151973670/377d8acc-e584-4c1d-93b7-2986bdb1c56a)

2. Neinteraktivni grafikoni:
Grafikon prosječnog broja kašnjenja za jedan mjesec po avio-kompaniji: ovaj grafikon prikazuje prosječan broj kašnjenja za jedan mjesec po avio-kompanijama. Koristi se trakasti grafikon.
![prosjek](https://github.com/bosiljkatodic00/BigDataProject/assets/151973670/1318106e-aa38-462c-bd74-741d57c6a301)

Takođe, omogućeno je generisanje izvještaja o aerodromu na osnovu izbora korisnika:

![izvjestaj](https://github.com/bosiljkatodic00/BigDataProject/assets/151973670/824f4649-3956-4895-a308-0e7a24968bc3)

-Predikcije:

Drugi skup podataka je korišćen za generisanje predikcija. Ulazne promjenljive su: X = df[['carrier_label', 'airport1_label', 'MONTH', 'DAY']],
a ciljna y = df['delayed'].
Testirala sam više klasifikacionih modela i dobila sljedeće rezultate:

Model: LogisticRegression

Training Time: 2.02 seconds

Accuracy: 0.50

Classification Report:

                   precision    recall  f1-score   support

               0       0.57      0.50      0.53    385082
               1       0.44      0.52      0.48    298184

        accuracy                           0.51    683266
       macro avg       0.51      0.51      0.50    683266
    weighted avg       0.52      0.51      0.51    683266

-----------------------
Model: RandomForestClassifier 

Training Time: 391.45 seconds

Accuracy: 0.59

Classification Report:

                   precision    recall  f1-score   support

               0       0.64      0.61      0.63    385082
               1       0.53      0.56      0.55    298184

        accuracy                           0.59    683266
       macro avg       0.59      0.59      0.59    683266
    weighted avg       0.59      0.59      0.59    683266

-----------------------
Model: XGBClassifier

Training Time: 5.43 seconds

Accuracy: 0.59

Classification Report:

                   precision    recall  f1-score   support

               0       0.65      0.59      0.62    385082
               1       0.53      0.59      0.56    298184

        accuracy                           0.59    683266
       macro avg       0.59      0.59      0.59    683266
    weighted avg       0.60      0.59      0.59    683266

-----------------------
Model: KNeighborsClassifier

Training Time: 4.81 seconds

Accuracy: 0.57

Classification Report:

                   precision    recall  f1-score   support

               0       0.62      0.64      0.63    385082
               1       0.51      0.48      0.50    298184

        accuracy                           0.57    683266
       macro avg       0.56      0.56      0.56    683266
    weighted avg       0.57      0.57      0.57    683266

-----------------------

Kao što možete vidjeti, XGBClassifier i RandomForestClassifier su se izjednačili kao najbolji modeli sa AUC od 0.59. 
Oba modela imaju slične performanse, ali XGBoost Classifier je postigao bolje rezultate sa manjim vremenom treniranja u poređenju sa Random Forest Classifier-om. 
Takođe, XGBoost Classifier ima nešto veći F1-score za klasu 0, dok su F1-score-ovi za klasu 1 približno jednaki za oba modela.

Dakle, na osnovu datih informacija, možemo reći da je XGBoost Classifier dao nešto bolje rezultate u poređenju sa Random Forest Classifier-om, uz značajno kraće vreme treniranja.

-Moguća unapređenja:

Da bismo unaprijedili rezultate predikcija, možemo razmotriti dodavanje dodatnih informacija u model. Nekoliko mogućih pristupa:

1. Uključivanje vremenskih podataka: Dodavanje informacija o trenutnom vremenu može biti korisno jer vremenski uslovi mogu uticati na kašnjenja letova.

2. Uključivanje podataka o stanju na pistama: Informacije o stanju na pistama, kao što su radovi na pistama, zatvaranja ili druge vazduhoplovne aktivnosti, takođe mogu biti korisne. 
