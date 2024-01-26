from flask import Flask, render_template, request
import pickle 

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) #read mode


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Logika za predikciju i dobijanje rezultata
    prediction_text = "Predikcija: ..."
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)