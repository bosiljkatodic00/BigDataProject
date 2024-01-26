from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
pd.options.mode.chained_assignment = None  # default='warn'

flight_data = pd.read_csv("Airline_Delay_Cause.csv")

df = flight_data.dropna()

label_encoder = LabelEncoder()

# Transformacija podataka u koloni 'carrier_name'
label = label_encoder.fit_transform(df['carrier_name'])

df['carrier_label'] = label
# Sada imate novu numeričku kolonu 'carrier_label' koju možete koristiti u vašem modelu
df.to_csv("C:/Users/Bosa/Desktop/BIg Data/Airline_Delay_Cause.csv", index=False)


# Izdvajanje ulaznih promenljivih (X) i ciljne promenljive (y)
X = df[['nas_ct', 'weather_ct', 'carrier_ct']]  # Promenljive koje će se koristiti za predikciju
y = df['arr_del15']  # Ciljna promenljiva (kašnjenje leta)

# Podela podataka na skup za obuku i skup za testiranje (obično se koristi 70-30 ili 80-20 podela)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Inicijalizacija modela
model = LinearRegression()

# Treniranje modela
model.fit(X_train, y_train)


# Predikcija na skupu za testiranje
y_pred = model.predict(X_test)

# Evaluacija performansi modela
mse = mean_squared_error(y_test, y_pred)  # Srednje kvadratno odstupanje
r2 = r2_score(y_test, y_pred)  # R-kvadrat vrednost

print(f"Srednje kvadratno odstupanje (MSE): {mse}")
print(f"R-kvadrat vrednost (R2): {r2}")

#cuvanje modela
file = open("model.pkl", 'wb')
pickle.dump(model, file)
