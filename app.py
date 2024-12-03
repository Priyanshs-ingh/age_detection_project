from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Load your trained model
model = pickle.load(open('final_life_expectancy_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[data['country'], data['age'], data['cigarettes_per_day']]])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
