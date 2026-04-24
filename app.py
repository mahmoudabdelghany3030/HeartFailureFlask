from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# تحميل الـ Pipeline الكامل (Scaler + Model مع بعض)
def load_artifacts():
    with open('heart_failure_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return pipeline, feature_names

pipeline, feature_names = load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # بناء DataFrame بنفس أسماء الأعمدة اللي اتدرب عليها الموديل
        # الترتيب: age, ejection_fraction, serum_creatinine, serum_sodium,
        #          high_blood_pressure, time, anaemia
        input_df = pd.DataFrame([[
            float(data['age']),
            float(data['ejection_fraction']),
            float(data['serum_creatinine']),
            float(data['serum_sodium']),
            int(data['high_blood_pressure']),
            float(data['time']),
            int(data['anaemia'])
        ]], columns=feature_names)

        prediction  = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0][1]

        return jsonify({
            'status':      'success',
            'prediction':  int(prediction[0]),
            'probability': round(float(probability), 2),
            'message':     'High Risk' if prediction[0] == 1 else 'Low Risk'
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
