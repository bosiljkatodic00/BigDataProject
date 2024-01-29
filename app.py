from flask import Flask, render_template, request, session
import pickle 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo

app = Flask(__name__)
app.secret_key = 'tk123321'

model = pickle.load(open('model.pkl','rb')) #read mode

# Ova funkcija će vam pomoći da konvertujete Matplotlib sliku u format pogodan za HTML
def plot_to_html_image(plt):
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"

# Učitavanje podataka
flight_data = pd.read_csv("Airline_Delay_Cause.csv")

# Uzimanje podataka samo za kašnjenja
delayed_flights = flight_data[flight_data['arr_del15'] > 15]

# Računanje prosečnog kašnjenja po prevozniku
avg_delay_by_carrier = delayed_flights.groupby('carrier_name')['arr_del15'].mean().reset_index()

# Uzimanje podataka samo o faktorima kašnjenja
delay_factors = flight_data[['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']]

# Konverzija godine i meseca u string format
flight_data['YearMonth'] = flight_data['year'].astype(str) + '-' + flight_data['month'].astype(str)

# Računanje ukupnog kašnjenja po mesecima i godinama
total_delay_by_month = flight_data.groupby('YearMonth')['arr_delay'].sum().reset_index()

# Uzimanje podataka samo za kašnjenja
delayed_flights_airports = flight_data[flight_data['arr_del15'] > 15]

# Računanje prosečnog kašnjenja po aerodromu
avg_delay_by_airport = delayed_flights_airports.groupby('airport_name')['arr_del15'].mean().reset_index()

# Grupisanje po avio-kompaniji
flights_by_carrier = flight_data.groupby('carrier_name')['arr_flights'].sum().reset_index()

fig = px.pie(flights_by_carrier, names='carrier_name', values='arr_flights', title='Broj letova po avio-prevozniku')

# Postavljanje opcija za prikaz informacija prilikom postavljanja kursora
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12)


# Računanje procentualnog udela otkazanih letova
flight_data['cancel_percentage'] = (flight_data['arr_cancelled'] / flight_data['arr_flights']) * 100

# Grupisanje po avio-kompaniji
cancel_percentage_by_carrier = flight_data.groupby('carrier_name')['cancel_percentage'].mean().reset_index()

# Kreiranje interaktivnog pie grafikona uz pomoć Plotly
fig = px.pie(cancel_percentage_by_carrier,
             names='carrier_name',
             values='cancel_percentage',
             title='Procentualni udio otkazanih letova po avio-prevozniku')

# Dodavanje stila za okrugle delove
fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)), selector=dict(type='pie'))

# Prikazivanje grafa
fig.write_html("plot.html")

plt.figure(figsize=(50, 50))
sns.barplot(x='arr_del15', y='airport_name', data=avg_delay_by_airport.sort_values(by='arr_del15', ascending=False))
plt.xlabel('Prosječni procenat kašnjenja')
plt.ylabel('Aerodrom')
plt.title('Prosječni procenat kašnjenja po aerodromu')
img4 = plot_to_html_image(plt)

# Generisanje grafikona unapred
plt.figure(figsize=(18, 10))
sns.barplot(x='arr_del15', y='carrier_name', data=avg_delay_by_carrier.sort_values(by='arr_del15', ascending=False))
plt.xlabel('Prosječni procenat kašnjenja')
plt.ylabel('Avio-prevoznik')
plt.title('Prosječni procenat kašnjenja po avio-kompaniji')
img1 = plot_to_html_image(plt)

plt.figure(figsize=(16, 6))
sns.boxplot(data=delay_factors)
plt.xlabel('Faktori kašnjenja')
plt.ylabel('Broj kašnjenja')
plt.title('Analiza faktora kašnjenja')
img2 = plot_to_html_image(plt)

plt.figure(figsize=(16, 6))
sns.lineplot(x='YearMonth', y='arr_delay', data=total_delay_by_month)
plt.xlabel('Godina-Mjesec')
plt.ylabel('Ukupno kašnjenje')
plt.title('Vremenska analiza ukupnog kašnjenja po mjesecima')
plt.xticks(rotation=45)
img3 = plot_to_html_image(plt)

# Sačuvajte generisane slike u sesiji
app.config['img1'] = img1
app.config['img2'] = img2
app.config['img3'] = img3


@app.route('/')
def index():
    return render_template('index.html', img1=img1, img2=img2, img3=img3)

@app.route('/airportcharts')
def airportcharts():
    return render_template('airportcharts.html', img4=img4)

@app.route('/carriercharts')
def carriercharts():
    return render_template('carriercharts.html', img1=img1)

@app.route('/predictdelay')
def predictdelay():
    return render_template('predictdelay.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Logika za predikciju i dobijanje rezultata
    # Prikupljanje parametara iz HTML forme
    month = int(request.form['month'])
    airport_label = int(request.form['airport_label']) 
    carrier_label = int(request.form['carrier_label'])  

    # Sačuvajte izabrane vrednosti u sesiji
    session['selected_month'] = request.form['month']
    session['selected_airport'] = request.form['airport_label']
    session['selected_carrier'] = request.form['carrier_label']

    # Pravljenje DataFrame-a sa podacima koje ćete dati modelu za predikciju
    input_data = pd.DataFrame({
        'carrier_label': [carrier_label],
        'airport_label': [airport_label],
        'month': [month]
    })

    # Predviđanje kašnjenja leta
    predicted_delay = model.predict(input_data)
    
    rounded_delay = round(predicted_delay[0])


    # Renderovanje HTML stranice sa predviđenim kašnjenjem
    return render_template('predictdelay.html', prediction_text=f'{rounded_delay} minuta',
                        selected_month=session.get('selected_month'),
                       selected_airport=session.get('selected_airport'),
                       selected_carrier=session.get('selected_carrier'))


if __name__ == '__main__':
    app.run(debug=True)