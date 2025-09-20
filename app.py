from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open('LinearRegressionModel.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
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
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(prediction, 2)})

if __name__ == '__main__':
    port = 5000
    # print(f"🚀 Server running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

