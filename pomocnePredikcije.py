from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Učitavanje podataka
flight_data = pd.read_csv("proba.csv", dtype={'ORIGIN_AIRPORT': str, 'DESTINATION_AIRPORT': str})
df = flight_data.dropna()

# Definisanje ciljne promenljive
df['delayed'] = df['DEPARTURE_DELAY'].apply(lambda x: 1 if x > 14 else 0)
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/proba2.csv", index=False)

label_encoder = LabelEncoder()

#label = label_encoder.fit_transform(df['AIRLINE'])

#df['carrier_label'] = label
# Sada imam novu numeričku kolonu 'carrier_label' 
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/proba.csv", index=False)

# Transformacija podataka u koloni 'carrier_name'
#label2 = label_encoder.fit_transform(df['ORIGIN_AIRPORT'])

#df['airport1_label'] = label2
# Sada imam novu numeričku kolonu 'airport_label' 
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/proba.csv", index=False)

# Transformacija podataka u koloni 'carrier_name'
#label3 = label_encoder.fit_transform(df['DESTINATION_AIRPORT'])

#df['airport2_label'] = label3
# Sada imam novu numeričku kolonu 'airport_label' 
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/proba.csv", index=False)

# Izdvajanje ulaznih promenljivih (X) i ciljne promenljive (y)
X = df[['carrier_label', 'airport1_label', 'MONTH', 'DAY']]
y = df['delayed']

# Podela podataka na skup za obuku i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicijalizacija Random Forest klasifikatora
rf_classifier = RandomForestClassifier(class_weight='balanced')

# Treniranje modela
rf_classifier.fit(X_train, y_train)

# Predikcija na testnom skupu
y_pred = rf_classifier.predict(X_test)

# Ispisivanje izveštaja klasifikacije
print("Classification Report:")
print(classification_report(y_test, y_pred))

#cuvanje modela
#file = open("modelFinal.pkl", 'wb')
#pickle.dump(rf_classifier, file)