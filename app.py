from flask import Flask, request, render_template,jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

# Load CSV data
data = pd.read_csv("aakash data - Sheet3.csv")

y1 =np.array([data['cases'].copy()])
y_mean=np.mean(y1)
y_std=np.std(y1)
y_new=(y1-y_mean)/y_std

# Create Flask app
app = Flask(__name__)
CORS(app)
# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
svr=pickle.load(open("svr.pkl", "rb"))

categorical_cols = ['state', 'mosquito']
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

@app.route("/", methods=["GET"])
def home():
    return jsonify({
       'prediction':10
    })

@app.route('/Prediction', methods=['POST'])
def predict():
    # Retrieve form data
    form_data = request.form

    # Convert form data into pandas DataFrame
    input_data = {
        'rainfall in mm': [float(form_data['feature2'])],
        'temperature': [float(form_data['feature3'])],
        'avg relative humidity': [float(form_data['feature4'])],
        'state': [form_data['feature5']],
        'mosquito': [form_data['feature6']]
    }
    df = pd.DataFrame(input_data)

    # Preprocess input data
    numeric_features = ['rainfall in mm', 'temperature', 'avg relative humidity']
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Encode categorical features
    encoded_data = encoder.transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)

    # Concatenate numeric and encoded categorical features
    df_combined = pd.concat([df[numeric_features], df_encoded], axis=1)

    # Make prediction
    no_of_cases=svr.predict(df_combined)
    no_of_cases=no_of_cases[0]
    no_of_cases=abs(int((no_of_cases*y_std)+y_mean))
    prediction = model.predict(df_combined)
    prediction=prediction[0]
    probability=model.predict_proba(df_combined)
    probability=np.max(probability)

    if no_of_cases>1500:
        cat="Category A"
    elif no_of_cases>800 and no_of_cases<1500:
        cat="Category B"
    elif no_of_cases>300 and no_of_cases<800:
        cat="Category C"
    else:
        cat="Category D"

    return jsonify({
        'prediction': prediction,
        'probability': probability,
        'no_of_cases': no_of_cases,
        'cat': cat
    })
@app.route("/predict_from_flutter", methods=["POST"])
def predict_from_flutter():
    # Retrieve JSON data sent from Flutter app
    data = request.json

    # Parse JSON data and extract input values
    feature2 = data['feature2']
    feature3 = data['feature3']
    feature4 = data['feature4']
    feature5 = data['feature5']
    feature6 = data['feature6']

    # Convert input values into appropriate data format
    input_data = {
        'rainfall in mm': [float(feature2)],
        'temperature': [float(feature3)],
        'avg relative humidity': [float(feature4)],
        'state': [feature5],
        'mosquito': [feature6]
    }
    df = pd.DataFrame(input_data)

    # Preprocess input data, make predictions, and prepare response
    # (Code similar to the /predict route above)
    numeric_features = ['rainfall in mm', 'temperature', 'avg relative humidity']
    df[numeric_features] = scaler.transform(df[numeric_features])
    encoded_data = encoder.transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)
    df_combined = pd.concat([df[numeric_features], df_encoded], axis=1)
    no_of_cases = svr.predict(df_combined)
    no_of_cases = no_of_cases[0]
    no_of_cases = abs(int((no_of_cases * y_std) + y_mean))
    prediction = model.predict(df_combined)
    prediction = prediction[0]
    probability = model.predict_proba(df_combined)
    probability = np.max(probability)

    if no_of_cases > 1500:
        cat = "Category A"
    elif no_of_cases > 800 and no_of_cases < 1500:
        cat = "Category B"
    elif no_of_cases > 300 and no_of_cases < 800:
        cat = "Category C"
    else:
        cat = "Category D"

    return jsonify({'prediction': prediction, 'probability': probability, 'no_of_cases': no_of_cases, 'category': cat})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
