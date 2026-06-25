from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)


# LOAD MODEL

with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']


# HOME PAGE

@app.route('/')
def home():
    return render_template('home.html')


# PREDICTION PAGE
@app.route('/predictor')
def predictor():

    locations = [
        col.replace('location_', '')
        for col in data_columns
        if col.startswith('location_')
    ]

    return render_template(
        'predictor.html',
        locations=sorted(locations)
    )


# PRICE PREDICTION API
@app.route('/predict', methods=['POST'])
def predict():

    try:
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        balcony = int(request.form['balcony'])
        location = request.form['location']

        x = np.zeros(len(data_columns))

        x[data_columns.index('total_sqft')] = total_sqft
        x[data_columns.index('bath')] = bath
        x[data_columns.index('BHK')] = bhk
        x[data_columns.index('balcony')] = balcony

        location_col = 'location_' + location

        if location_col in data_columns:
            x[data_columns.index(location_col)] = 1

        prediction = model.predict([x])[0]

        
        prediction = np.exp(prediction)

        return jsonify({
            'success': True,
            'price': round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'error': str(e)
        })


# RUN APP

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))