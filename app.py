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

model = pickle.load(open('modelFinal.pkl','rb')) #read mode

# Učitavanje podataka
flight_data = pd.read_csv("Airline_Delay_Cause.csv")


#OVO JE ZA GENERISANJE HTML STANICA KOJE CE BITI INTERAKTIVNI DIJAGRAMI
# Grupisanje po avio-kompaniji
flights_by_carrier = flight_data.groupby('carrier_name')['arr_flights'].sum().reset_index()
fig = px.pie(flights_by_carrier, names='carrier_name', values='arr_flights', title='Broj letova po avio-kompaniji')
# Postavljanje opcija za prikaz informacija prilikom postavljanja kursora
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12)
#fig.write_html("dijagram.html")

# Računanje procentualnog udela otkazanih letova
flight_data['cancel_percentage'] = (flight_data['arr_cancelled'] / flight_data['arr_flights']) * 100
# Grupisanje po avio-kompaniji
cancel_percentage_by_carrier = flight_data.groupby('carrier_name')['cancel_percentage'].mean().reset_index()
# Kreiranje interaktivnog pie grafikona uz pomoć Plotly
fig = px.pie(cancel_percentage_by_carrier,
             names='carrier_name',
             values='cancel_percentage',
             title='Procentualni udio otkazanih letova po avio-kompaniji')
# Dodavanje stila za okrugle delove
fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)), selector=dict(type='pie'))
#fig.write_html("plot.html")


#OVO JE ZA GRAFIKON KOJI NIJE INTERAKTIVAN
# Ova funkcija će da konvertuje Matplotlib sliku u format pogodan za HTML
def plot_to_html_image(plt):
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"


# Uzimanje podataka samo za kašnjenja
delayed_flights = flight_data[flight_data['arr_del15'] > 15]

# Računanje prosečnog kašnjenja po kompaniji
avg_delay_by_carrier = delayed_flights.groupby('carrier_name')['arr_del15'].mean().reset_index()

# Generisanje grafikona unapred
plt.figure(figsize=(18, 10))
sns.barplot(x='arr_del15', y='carrier_name', data=avg_delay_by_carrier.sort_values(by='arr_del15', ascending=False))
plt.xlabel('Prosječni broj kašnjenja za jedan mjesec')
plt.ylabel('Avio-kompanija')
plt.title('Prosječni broj kašnjenja za jedan mjesec po avio-kompaniji')
img1 = plot_to_html_image(plt)
# Sačuvajte generisane slike u sesiji
app.config['img1'] = img1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/airportcharts')
def airportcharts():
    return render_template('airportcharts.html')

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
    day = request.form['dayOfMonth']
    if day:
        day = int(day)
    else:
        day = 1
    airport_label = int(request.form['airport_label']) 
    carrier_label = int(request.form['carrier_label'])  

    session['selected_month'] = request.form['month']
    session['selected_day'] = request.form['dayOfMonth']
    session['selected_airport'] = request.form['airport_label']
    session['selected_carrier'] = request.form['carrier_label']

    # Pravljenje DataFrame-a sa podacima za predikciju
    input_data = pd.DataFrame({
        'carrier_label': [carrier_label],
        'airport1_label': [airport_label],
        'MONTH': [month], 
        'DAY' : [day]
    })

    # Predviđanje kašnjenja leta
    predicted_probabilities = model.predict_proba(input_data)

    # Vjerovatnoća kašnjenja leta
    delay_probability = predicted_probabilities[0][1]  # Vjerovatnoća klase 1 (kašnjenje)
    delay_percent = round(delay_probability * 100, 2)  # Pretvaranje u procenat sa dve decimale

    # Da li let kasni ili ne
    if delay_probability >= 0.5:
        is_delayed = 'Da'
    else:
        is_delayed = 'Ne'

    # Renderovanje HTML stranice sa predviđenim kašnjenjem
    return render_template('predictdelay.html', prediction_text=f'Vjerovatnoća kašnjenja: {delay_percent}% <br>Let kasni: {is_delayed}',
                        selected_month=session.get('selected_month'),
                        selected_day=session.get('selected_day'),
                       selected_airport=session.get('selected_airport'),
                       selected_carrier=session.get('selected_carrier'))


# Ruta za obrađivanje izbora aerodroma i prikaz rezultata
@app.route('/rank_airport', methods=['GET', 'POST'])
def rank_airport():
   
    airport = request.form['airport_name']
    
    aerodrom_podaci = flight_data[flight_data['airport_name'] == airport]
    poruka='.'
    if aerodrom_podaci.empty:
            return render_template('airportcharts.html', poruka='Aerodrom nije pronađen.')

    prosecno_kasnjenje = (aerodrom_podaci['arr_delay'].mean()/aerodrom_podaci['arr_del15'].mean())
    otkazani = aerodrom_podaci['arr_cancelled'].sum()
    brojLetova = aerodrom_podaci['arr_flights'].sum()

    return render_template('airportcharts.html', airport=airport, prosecno_kasnjenje=round(prosecno_kasnjenje), poruka=poruka, brojLetova = round(brojLetova), otkazani = round(otkazani), aerodrom_podaci = aerodrom_podaci)


if __name__ == '__main__':
    app.run(debug=True)