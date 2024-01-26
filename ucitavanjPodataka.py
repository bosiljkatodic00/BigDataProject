import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

flight_data = pd.read_csv("Airline_Delay_Cause.csv")

# Uzimanje podataka samo za kašnjenja
delayed_flights = flight_data[flight_data['arr_del15'] > 15]

# Računanje prosečnog kašnjenja po prevozniku
avg_delay_by_carrier = delayed_flights.groupby('carrier_name')['arr_del15'].mean().reset_index()

# Prikazivanje grafikona
plt.figure(figsize=(12, 6))
sns.barplot(x='arr_del15', y='carrier_name', data=avg_delay_by_carrier.sort_values(by='arr_del15', ascending=False))
plt.xlabel('Prosečni procenat kašnjenja')
plt.ylabel('Avio-prevoznik')
plt.title('Prosečni procenat kašnjenja po avio-prevozniku')
#plt.show()

# Uzimanje podataka samo o faktorima kašnjenja
delay_factors = flight_data[['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']]

# Prikazivanje boxplot-a za analizu faktora kašnjenja
plt.figure(figsize=(10, 6))
sns.boxplot(data=delay_factors)
plt.xlabel('Faktori kašnjenja')
plt.ylabel('Broj kašnjenja')
plt.title('Analiza faktora kašnjenja')
#plt.show()

# Konverzija godine i meseca u string format
flight_data['YearMonth'] = flight_data['year'].astype(str) + '-' + flight_data['month'].astype(str)

# Računanje ukupnog kašnjenja po mesecima i godinama
total_delay_by_month = flight_data.groupby('YearMonth')['arr_delay'].sum().reset_index()

# Prikazivanje linčog grafikona za ukupno kašnjenje tokom vremena
plt.figure(figsize=(12, 6))
sns.lineplot(x='YearMonth', y='arr_delay', data=total_delay_by_month)
plt.xlabel('Godina-Mesec')
plt.ylabel('Ukupno kašnjenje')
plt.title('Vremenska analiza ukupnog kašnjenja')
plt.xticks(rotation=45)
plt.show()
