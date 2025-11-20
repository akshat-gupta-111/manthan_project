from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join('model', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [float(x) for x in data.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return jsonify({'prediction': output})

if __name__ == "__main__":
    app.run(debug=True)