from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

flight_data = pd.read_csv("Airline_Delay_Cause.csv")

df = flight_data.dropna()

label_encoder = LabelEncoder()

# Transformacija podataka u koloni 'carrier_name'
#label = label_encoder.fit_transform(df['carrier_name'])

#df['carrier_label'] = label
# Sada imam novu numeričku kolonu 'carrier_label' 
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/Airline_Delay_Cause.csv", index=False)

# Transformacija podataka u koloni 'carrier_name'
#label2 = label_encoder.fit_transform(df['airport_name'])

#df['airport_label'] = label2
# Sada imam novu numeričku kolonu 'airport_label' 
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/Airline_Delay_Cause.csv", index=False)

df['target'] = np.where((df['arr_del15'] != 0) & (df['arr_delay'] != 0), df['arr_delay'] / df['arr_del15'], 0)
df['target'] = df['target'].round(2)
#df.to_csv("C:/Users/Bosa/Desktop/BIg Data/Airline_Delay_Cause.csv", index=False)

# Izdvajanje ulaznih promenljivih (X) i ciljne promenljive (y)
X = df[['carrier_label', 'airport_label', 'month']]  # Promenljive koje će se koristiti za predikciju
y = df['target']   # Ciljna promenljiva 

# Podela podataka na skup za obuku i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()

# Treniranje modela
model.fit(X_train, y_train)

# Predikcija na skupu za testiranje
y_pred = model.predict(X_test)

# Grafikon stvarnih vrednosti i predviđenih vrednosti
plt.scatter(y_test, y_pred)
plt.xlabel("Stvarna kašnjenja")
plt.ylabel("Predviđena kašnjenja")
plt.title("Stvarna vs. Predviđena kašnjenja")
plt.show()

# Evaluacija performansi modela
mse = mean_squared_error(y_test, y_pred)  # Srednje kvadratno odstupanje
r2 = r2_score(y_test, y_pred)  # R-kvadrat vrednost

print(f"Srednje kvadratno odstupanje (MSE): {mse}")
print(f"R-kvadrat vrednost (R2): {r2}")

#cuvanje modela
file = open("model2.pkl", 'wb')
pickle.dump(model, file)
