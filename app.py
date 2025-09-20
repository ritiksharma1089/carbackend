import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Allow CORS for all domains (for development; restrict in production)
# Load trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON like:
    {
        "company": "Maruti",
        "car_model": "Swift",
        "year": 2019,
        "fuel_type": "Petrol",
        "kms_driven": 10000
    }
    """
    data = request.get_json()

    # Create DataFrame for prediction
    input_df = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([
            data['car_model'],
            data['company'],
            data['year'],
            data['kms_driven'],
            data['fuel_type']
        ]).reshape(1, 5)
    )

    # Predict
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(prediction, 2)})

if __name__ == '__main__':
    # Use Railway PORT or default 5000
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running at http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
