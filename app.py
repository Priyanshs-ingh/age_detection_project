from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)
model = load('health_index_model.joblib')  # Load your trained model from the file

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Parse inputs for Disease15 and Smoking_2021
            disease_metric = float(request.form['disease_metric'])
            smoking_rate = float(request.form['smoking_rate'])
            # Prepare the DataFrame in the same order as the model expects
            features_df = pd.DataFrame([[disease_metric, smoking_rate]])
            prediction = model.predict(features_df)
            prediction_text = f'Predicted Health Index: {prediction[0]:.2f}'
        except ValueError:
            prediction_text = "Please enter valid numbers."
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
