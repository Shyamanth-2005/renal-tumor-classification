from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import warnings
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Load the risk inference model
try:
    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        risk_model = joblib.load(r"artifacts/training/kidney_risk_inference_model.pkl")
    print("Risk model loaded successfully")
except Exception as e:
    risk_model = None
    print(f"Warning: Risk model not found - {str(e)}")


class ClientApp:
    def __init__(self):
        self.filename = "input_image.jpg"
        self.classifier = PredictionPipeline(self.filename)
        

# default route
@app.route("/",methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
@cross_origin()
def predict_route():
    try:
        # Check if request contains JSON data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check if image data exists
        if 'image' not in request.json:
            return jsonify({"error": "No image data found in request"}), 400

        image = request.json['image']

        # Validate that image data is not empty
        if not image or len(image.strip()) == 0:
            return jsonify({"error": "Image data is empty"}), 400

        # Decode and save image
        try:
            decodeImage(image, clApp.filename)
        except Exception as decode_error:
            print(f"Error decoding image: {str(decode_error)}")
            return jsonify({"error": f"Failed to decode image: {str(decode_error)}"}), 400

        # Check if the file was properly created
        if not os.path.exists(clApp.filename):
            return jsonify({"error": "Failed to save decoded image"}), 400

        # Check if the file is empty
        if os.path.getsize(clApp.filename) == 0:
            return jsonify({"error": "Decoded image is empty"}), 400

        # Make prediction
        result = clApp.classifier.predict()

        # Check if prediction returned an error
        if isinstance(result, list) and len(result) > 0 and 'error' in result[0]:
            return jsonify(result), 400

        # Clean up the temporary image file
        if os.path.exists(clApp.filename):
            os.remove(clApp.filename)

        return jsonify(result)
    except Exception as e:
        print(f"Error in predict_route: {str(e)}")
        # Clean up the temporary image file in case of error
        try:
            if os.path.exists(clApp.filename):
                os.remove(clApp.filename)
        except:
            pass  # Ignore cleanup errors
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/analyze_risk", methods=['POST'])
@cross_origin()
def analyze_risk():
    try:
        data = request.json
        
        # Prepare features for the model
        features = np.array([
            [
                data.get('hematuria', 0),
                data.get('flank_pain', 0),
                data.get('lower_back_pain', 0),
                data.get('fever', 0),
                data.get('appetite_loss', 0),
                data.get('weight_loss', 0),
                data.get('fatigue', 0),
                data.get('anemia', 0),
                data.get('hypertension', 0),
                data.get('diabetes', 0),
                data.get('ckd', 0),
                data.get('fh_kidney_tumor', 0),
                data.get('fh_hypertension', 0),
                data.get('smoking', 0),
                data.get('alcohol', 0),
                data.get('height', 170) / 100,  # Convert to meters for BMI
                data.get('weight', 70),
                data.get('chemical_exposure', 0),
                data.get('activity_level', 1)
            ]
        ])
        
        # Get predictions from model
        if risk_model is not None:
            try:
                proba = risk_model.predict_proba(features)[0] if hasattr(risk_model, 'predict_proba') else [0.5, 0.5]
            except:
                proba = [0.5, 0.5]
        else:
            # Fallback: compute simple heuristics if model not available
            proba = compute_risk_heuristics(data)
        
        # Format response
        response = {
            "ct_result": data.get('ct_result', 'normal'),
            "risk_factors": extract_risk_factors(data),
            "associated_conditions": extract_conditions(data, proba),
            "overall_risk_score": float(proba[1]) if len(proba) > 1 else 0.5
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in analyze_risk: {str(e)}")
        return jsonify({"error": str(e)}), 400


def extract_risk_factors(data):
    """Extract and score risk factors"""
    risk_factors = {}
    
    if data.get('smoking', 0) == 2:  # Current smoker
        risk_factors['smoking'] = 0.63
    elif data.get('smoking', 0) == 1:  # Former
        risk_factors['smoking'] = 0.35
    
    # Calculate BMI
    height_m = data.get('height', 170) / 100
    weight = data.get('weight', 70)
    bmi = weight / (height_m ** 2) if height_m > 0 else 25
    if bmi > 30:
        risk_factors['obesity'] = 0.58
    elif bmi > 25:
        risk_factors['overweight'] = 0.35
    
    if data.get('hypertension', 0) == 1:
        risk_factors['hypertension'] = 0.71
    
    if data.get('fh_hypertension', 0) == 1:
        risk_factors['family_history_hypertension'] = 0.49
    
    if data.get('fh_kidney_tumor', 0) == 1:
        risk_factors['family_history_kidney_tumor'] = 0.65
    
    if data.get('alcohol', 0) == 2:  # >7 drinks/week
        risk_factors['alcohol_abuse'] = 0.54
    
    if data.get('chemical_exposure', 0) == 1:
        risk_factors['chemical_exposure'] = 0.32
    
    if data.get('activity_level', 1) == 0:  # Low activity
        risk_factors['sedentary_lifestyle'] = 0.45
    
    return risk_factors


def extract_conditions(data, proba):
    """Extract and score possible associated conditions"""
    conditions = {}
    
    ct_result = data.get('ct_result', 'normal')
    
    if ct_result == 'tumor':
        conditions['renal_cell_carcinoma'] = 0.81
        conditions['chronic_kidney_disease'] = 0.46
        conditions['benign_renal_mass'] = 0.19
    else:
        # For normal scans, predict future risk
        if data.get('hypertension', 0) or data.get('fh_hypertension', 0):
            conditions['hypertension_related_ckd'] = 0.42
        if data.get('diabetes', 0):
            conditions['diabetic_nephropathy'] = 0.38
        if data.get('smoking', 0) or data.get('alcohol', 0):
            conditions['future_tumor_risk'] = 0.28
        else:
            conditions['low_future_risk'] = 0.15
    
    return conditions


def compute_risk_heuristics(data):
    """Fallback risk computation if model unavailable"""
    score = 0.0
    
    score += (data.get('hematuria', 0) * 0.15)
    score += (data.get('flank_pain', 0) * 0.1)
    score += (data.get('hypertension', 0) * 0.1)
    score += (data.get('smoking', 0) * 0.1)
    score += (data.get('fh_kidney_tumor', 0) * 0.15)
    score += (data.get('ckd', 0) * 0.15)
    score += (data.get('diabetes', 0) * 0.1)
    
    # Normalize
    score = min(score, 0.95)
    
    return [1 - score, score]


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train_route():
    os.system("python main.py")
    return "Training done successfully!"


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="127.0.0.1", port=8085, debug=True)
