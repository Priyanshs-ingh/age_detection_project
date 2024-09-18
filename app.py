from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the model
model = load('optimized_gbm.pkl')

@app.route('/')
def home():
    return render_template('age_prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data sent from the frontend
    data = request.get_json(force=True)
    
    # Assuming 'smoker_status' and 'age' are correctly sent by the frontend
    smoker_status = data['smoker_status']
    age = data['age']

    # Convert data into the correct format for your model
    # Assuming your model needs an array of features: [smoker_status, age]
    # You might need to convert smoker_status to a numerical format if your model expects it
    prediction_input = np.array([[smoker_status, age]])

    # Generate predictions
    prediction = model.predict(prediction_input)

    # Send back the prediction in JSON format
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
