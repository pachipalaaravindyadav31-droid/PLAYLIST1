import os
import csv
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

from src.utils import get_path, get_logger
from src.data_preprocessing import preprocess_features
from src.dataset_manager import (
    get_active_dataset_path,
    get_dataset_summary,
    save_and_activate_dataset,
    list_all_datasets,
    set_active_dataset
)
from src.train_all import train_everything


logger = get_logger(__name__)

app = Flask(__name__)

# Cache variables for models & metadata
METRICS = None
SCALER = None
ENCODER = None
BASELINE_MEANS = {}
LOG_FILE_PATH = get_path('logs', 'predictions.csv')

# Bounding boxes for schema validation
FEATURE_LIMITS = {
    'duration_ms': (1000, 3600000),      # 1 sec to 1 hour
    'danceability': (0.0, 1.0),
    'energy': (0.0, 1.0),
    'key': (0, 11),
    'loudness': (-60.0, 5.0),
    'mode': (0, 1),
    'speechiness': (0.0, 1.0),
    'acousticness': (0.0, 1.0),
    'instrumentalness': (0.0, 1.0),
    'liveness': (0.0, 1.0),
    'valence': (0.0, 1.0),
    'tempo': (0.0, 300.0),
    'time_signature': (3, 7)
}

def init_app_resources():
    """Initializes and caches models, encoders, and baseline statistics on startup."""
    global METRICS, SCALER, ENCODER, BASELINE_MEANS
    
    # 1. Load metrics summary
    metrics_path = get_path('models', 'metrics_summary.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            METRICS = json.load(f)
        logger.info("Loaded metrics summary successfully.")
    else:
        logger.error(f"No metrics summary found at {metrics_path}. Please run offline training first.")
        
    # 2. Load preprocessors
    scaler_path = get_path('models', 'scaler.joblib')
    encoder_path = get_path('models', 'encoder.joblib')
    if os.path.exists(scaler_path) and os.path.exists(encoder_path):
        SCALER = joblib.load(scaler_path)
        ENCODER = joblib.load(encoder_path)
        logger.info("Loaded fitted StandardScaler and OneHotEncoder successfully.")
    else:
        logger.error("Preprocessors not found. Run training pipeline.")
        
    # 3. Calculate baseline means from clean sample data for drift monitoring
    sample_path = get_path('data', 'clean_sample.csv')
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        for col in ['danceability', 'energy', 'acousticness', 'speechiness']:
            BASELINE_MEANS[col] = float(df_sample[col].mean())
        logger.info(f"Loaded baseline feature means: {BASELINE_MEANS}")
    else:
        # Fallback values
        BASELINE_MEANS = {
            'danceability': 0.566,
            'energy': 0.641,
            'acousticness': 0.314,
            'speechiness': 0.084
        }
        logger.info(f"Using fallback baseline means: {BASELINE_MEANS}")
        
    # 4. Initialize prediction logs CSV file
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                'time_signature', 'explicit', 'track_genre', 'regression_model',
                'classification_model', 'predicted_popularity', 'predicted_skip_probability',
                'skip_risk_category', 'model_version'
            ])
        logger.info(f"Initialized new prediction log file at {LOG_FILE_PATH}")

# Initialize at module import
init_app_resources()

def validate_features(feat_dict):
    """Validates types and ranges of input audio features."""
    # Check completeness
    for f in FEATURE_LIMITS:
        if f not in feat_dict:
            return False, f"Missing required feature: '{f}'"
            
    if 'explicit' not in feat_dict:
        return False, "Missing required feature: 'explicit'"
    if 'track_genre' not in feat_dict:
        return False, "Missing required feature: 'track_genre'"
        
    # Check numeric ranges
    for feature, (min_val, max_val) in FEATURE_LIMITS.items():
        val = feat_dict[feature]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return False, f"Feature '{feature}' must be a numeric value."
            
        if not (min_val <= val_float <= max_val):
            return False, f"Feature '{feature}' value {val_float} is out of bounds ({min_val} to {max_val})."
            
    # Check explicit (bool)
    if not isinstance(feat_dict['explicit'], bool):
        return False, "Feature 'explicit' must be a boolean."
        
    # Check track_genre (str)
    if not isinstance(feat_dict['track_genre'], str) or not feat_dict['track_genre'].strip():
        return False, "Feature 'track_genre' must be a non-empty string."
        
    return True, None

def get_recent_logs(n=8):
    """Retrieves last n records from predictions log file."""
    if not os.path.exists(LOG_FILE_PATH):
        return [], 0
        
    logs = []
    total_count = 0
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_count = len(rows)
            # Get last n rows
            logs = rows[-n:][::-1] # reverse to show newest first
    except Exception as e:
        logger.error(f"Error reading prediction logs: {e}")
        
    return logs, total_count

def calculate_drift_metrics():
    """Calculates data drift metrics comparing baseline training means to recent predictions."""
    if not os.path.exists(LOG_FILE_PATH):
        return {}, False, 0, 0.0, 0.0
        
    drift_metrics = {}
    drift_flag = False
    
    try:
        df = pd.read_csv(LOG_FILE_PATH)
        log_count = len(df)
        
        # If there are no logged predictions, return empty drift metrics
        if log_count == 0:
            for feat, base_val in BASELINE_MEANS.items():
                drift_metrics[feat] = {
                    'baseline': base_val,
                    'logged': base_val,
                    'deviation': 0.0,
                    'drift_detected': False
                }
            return drift_metrics, False, 0, 33.2, 0.206
            
        # Compute recent averages (use last 50 predictions, or all if less than 50)
        window = df.tail(50)
        
        avg_pop = float(df['predicted_popularity'].mean()) if 'predicted_popularity' in df.columns else 33.2
        
        # Average skip prob
        avg_skip_prob = float(df['predicted_skip_probability'].mean()) if 'predicted_skip_probability' in df.columns else 0.206
        
        for feature, baseline_val in BASELINE_MEANS.items():
            if feature in window.columns:
                logged_val = float(window[feature].mean())
            else:
                logged_val = baseline_val
                
            deviation = logged_val - baseline_val
            # Flag drift if difference exceeds absolute threshold of 0.10
            drift_detected = abs(deviation) >= 0.10
            
            if drift_detected:
                drift_flag = True
                
            drift_metrics[feature] = {
                'baseline': baseline_val,
                'logged': logged_val,
                'deviation': deviation,
                'drift_detected': drift_detected
            }
            
        return drift_metrics, drift_flag, log_count, avg_pop, avg_skip_prob
        
    except Exception as e:
        logger.error(f"Error calculating drift metrics: {e}")
        # Return fallbacks
        return {}, False, 0, 33.2, 0.206

# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    """Renders the dashboard homepage with summary metrics and model comparisons."""
    if METRICS is None:
        return "Model training has not been run. Please run 'python -m src.train_all' first.", 500
        
    reg_metrics = METRICS['regression']['comparison']
    # Sort regression metrics by test R2 in descending order
    reg_metrics_sorted = sorted(reg_metrics.items(), key=lambda x: x[1]['test']['r2'], reverse=True)
    
    best_reg_name = METRICS['regression']['best_model']
    
    clf_metrics = METRICS['classification']['binary_comparison']
    # Sort classification metrics by test F1 in descending order
    clf_metrics_sorted = sorted(clf_metrics.items(), key=lambda x: x[1]['test']['f1'], reverse=True)
    
    best_clf_name = METRICS['classification']['best_binary_model']
    multi_metrics = METRICS['classification']['multinomial_results']
    
    return render_template(
        'index.html',
        reg_metrics=reg_metrics_sorted,
        best_reg_name=best_reg_name,
        clf_metrics=clf_metrics_sorted,
        best_clf_name=best_clf_name,
        multi_metrics=multi_metrics
    )

@app.route('/eda')
def eda():
    """Renders the Exploratory Data Analysis (EDA) dashboard."""
    return render_template('eda.html')

@app.route('/experiments')
def experiments():
    """Renders the Hyperparameter sweeps, decision tree splits, and scaling sensitivity plots."""
    if METRICS is None:
        return "Model metrics summary missing.", 500
        
    scaling_metrics = METRICS['regression']['experiments']['scaling']
    return render_template('experiments.html', scaling_metrics=scaling_metrics)

@app.route('/concepts')
def concepts():
    """Renders the comprehensive M1, M2, M3, M4, and M5 ML concept curriculum breakdown."""
    return render_template('concepts.html')

@app.route('/graphs')
def graphs():
    """Renders the dedicated visual Concept Graphs studio for Modules M1-M5."""
    return render_template('graphs.html')


@app.route('/predict')
def predict_page():
    """Renders the interactive inference playground form."""
    genres = ['acoustic']
    if ENCODER is not None:
        try:
            genres = sorted(ENCODER.categories_[0].tolist())
        except Exception:
            pass
    return render_template('predict.html', genres=genres)

@app.route('/monitoring')
def monitoring():
    """Renders the database logs panel and drift analyzer."""
    recent_logs, _ = get_recent_logs(8)
    drift_metrics, drift_flag, log_count, avg_pop, avg_skip_prob = calculate_drift_metrics()
    
    return render_template(
        'monitoring.html',
        logs=recent_logs,
        drift_metrics=drift_metrics,
        drift_flag=drift_flag,
        log_count=log_count,
        avg_popularity=avg_pop,
        avg_skip_prob=avg_skip_prob
    )

@app.route('/dataset')
def dataset_page():
    """Renders the Dataset Management Hub and online retraining studio."""
    summary = get_dataset_summary()
    datasets, active_path = list_all_datasets()
    return render_template('dataset.html', summary=summary, datasets=datasets, active_path=active_path)

# ==================== DATASET & RETRAINING APIS ====================

@app.route('/api/dataset/upload', methods=['POST'])
def upload_dataset_api():
    """Uploads a new CSV dataset, validates schema, and registers it."""
    if 'dataset_file' not in request.files:
        return jsonify({'success': False, 'error': "No file part in the request."}), 400
        
    file = request.files['dataset_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': "No file selected for uploading."}), 400
        
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'success': False, 'error': "Only .csv format files are accepted."}), 400
        
    make_active = request.form.get('make_active', 'true').lower() == 'true'
    
    success, message, summary = save_and_activate_dataset(file, file.filename, make_active=make_active)
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'summary': summary
        })
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/api/dataset/switch', methods=['POST'])
def switch_dataset_api():
    """Switches the active dataset to a previously registered dataset."""
    req_data = request.get_json(silent=True) or {}
    dataset_path = req_data.get('dataset_path')
    if not dataset_path:
        return jsonify({'success': False, 'error': "Missing 'dataset_path' parameter."}), 400
        
    success, message = set_active_dataset(dataset_path)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/api/dataset/retrain', methods=['POST'])
def retrain_pipeline_api():
    """Triggers the full ML training pipeline (M1-M5) on the currently active dataset."""
    try:
        req_data = request.get_json(silent=True) or {}
        sample_size = int(req_data.get('sample_size', 15000))
        active_dataset = get_active_dataset_path()
        
        logger.info(f"Triggering automated ML retraining on: {active_dataset} (Sample size: {sample_size})")
        summary = train_everything(sample_size=sample_size, filepath=active_dataset)
        
        # Reload app in-memory caches
        init_app_resources()
        
        return jsonify({
            'success': True,
            'message': "End-to-end ML pipeline (M1 - M5) executed and models reloaded successfully!",
            'model_version': summary.get('model_version', '2.0.0')
        })
    except Exception as e:
        logger.error(f"Retraining error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== API ENDPOINT ====================


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST API to predict popularity and skip probability.
    Expects JSON payload with 'features', 'regression_model', and 'classification_model'.
    """
    # 1. Parse request JSON
    req_data = request.get_json(silent=True)
    if not req_data or 'features' not in req_data:
        return jsonify({'error': "Invalid request format. Must contain a 'features' dictionary."}), 400
        
    features = req_data['features']
    reg_model_name = req_data.get('regression_model', 'best_model')
    clf_model_name = req_data.get('classification_model', 'best_model')
    
    # 2. Input Validation
    is_valid, err_msg = validate_features(features)
    if not is_valid:
        return jsonify({'error': f"Input Validation Error: {err_msg}"}), 400
        
    # 3. Model path resolution
    reg_model_file = 'best_model.joblib' if reg_model_name == 'best_model' else f'{reg_model_name}.joblib'
    clf_model_file = 'best_model.joblib' if clf_model_name == 'best_model' else f'{clf_model_name}.joblib'
    
    reg_path = get_path('models', 'regression', reg_model_file)
    clf_path = get_path('models', 'classification', clf_model_file)
    
    if not os.path.exists(reg_path) or not os.path.exists(clf_path):
        return jsonify({'error': "Selected machine learning models are missing on disk."}), 500
        
    try:
        # 4. Load Models & Pipelines
        reg_model = joblib.load(reg_path)
        clf_model = joblib.load(clf_path)
        
        # 5. Preprocessing & Alignment
        # Convert single input dict to DataFrame
        input_df = pd.DataFrame([features])
        
        # Split into numerical feature columns used in preprocessing
        num_cols = ['duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'time_signature']
        
        # Transform using StandardScaler and OneHotEncoder
        X_input, _ = preprocess_features(input_df, SCALER, ENCODER, num_cols, scaled=True)
        
        # 6. Run Inference
        pred_pop = float(reg_model.predict(X_input)[0])
        pred_pop = np.clip(pred_pop, 0.0, 100.0) # clip to popularity limits
        
        # Predict skip probability (binary classifier)
        pred_skip_prob = float(clf_model.predict_proba(X_input)[0][1])
        
        # 7. Map Skip Risk Category from probability thresholds
        if pred_skip_prob < 0.35:
            risk_category = 'Low'
        elif pred_skip_prob < 0.60:
            risk_category = 'Medium'
        else:
            risk_category = 'High'
            
        timestamp = datetime.datetime.now().isoformat()
        model_version = '1.0.0'
        
        # 8. Log prediction request to CSV file
        try:
            with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, features['duration_ms'], features['danceability'],
                    features['energy'], features['key'], features['loudness'], features['mode'],
                    features['speechiness'], features['acousticness'], features['instrumentalness'],
                    features['liveness'], features['valence'], features['tempo'],
                    features['time_signature'], features['explicit'], features['track_genre'],
                    reg_model_name, clf_model_name, pred_pop, pred_skip_prob, risk_category,
                    model_version
                ])
        except Exception as le:
            logger.error(f"Failed to log prediction to file: {le}")
            
        # 9. Return JSON payload
        response = {
            'popularity_prediction': pred_pop,
            'skip_prediction_probability': pred_skip_prob,
            'skip_risk_category': risk_category,
            'model_version': model_version,
            'models_used': {
                'regression': reg_model_name,
                'classification': clf_model_name
            }
        }
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Inference pipeline execution failure: {e}", exc_info=True)
        return jsonify({'error': f"Inference execution failure: {str(e)}"}), 500

if __name__ == '__main__':
    # Start flask application
    logger.info("Starting PlaylistPulse Flask server on http://127.0.0.1:5000...")
    app.run(debug=True, host='127.0.0.1', port=5000)
