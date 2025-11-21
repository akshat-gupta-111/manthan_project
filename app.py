from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open('model/model.pkl', 'rb'))
encoders = pickle.load(open('model/encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 

    try:
        data = request.json
        
        
        age = float(data['age_months'])
        weight = float(data['weight_kg'])
        muac = float(data['muac_cm'])
        
        
        dehydration = encoders['dehydration_grade'][data['dehydration_grade']]
        pathogen = encoders['pathogen_identified'][data['pathogen_identified']]
        treatment = encoders['treatment_group'][data['treatment_group']]
        
        
        resistance = int(data['azithro_resistance_detected'])
        
        
        features = np.array([[age, weight, muac, dehydration, pathogen, resistance, treatment]])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        result_text = "Rapid Recovery Likely (<48hrs)" if prediction == 1 else "Slow Recovery Expected (>48hrs)"
        
        return jsonify({
            'prediction_text': result_text,
            'probability': f"{probability*100}% confidence"
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    app.run(debug=True)