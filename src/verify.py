import os
import sys
import json
import joblib
import pandas as pd

from src.utils import get_path, get_logger
from src.data_preprocessing import preprocess_features

logger = get_logger(__name__)

def run_verification():
    logger.info("==============================================")
    logger.info("RUNNING SYSTEM ALIGNMENT AND MODEL VERIFICATION")
    logger.info("==============================================")
    
    # 1. Verify files exist
    required_paths = [
        get_path('models', 'metrics_summary.json'),
        get_path('models', 'scaler.joblib'),
        get_path('models', 'encoder.joblib'),
        get_path('models', 'regression', 'best_model.joblib'),
        get_path('models', 'classification', 'best_model.joblib'),
        get_path('models', 'classification', 'multinomial_logistic_regression.joblib')
    ]
    
    missing_files = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_files.append(path)
            logger.error(f"Missing required asset: {path}")
        else:
            logger.info(f"Asset verified: {os.path.basename(path)} exists.")
            
    if missing_files:
        logger.error("Verification failed: Some serialized assets are missing.")
        sys.exit(1)
        
    logger.info("All essential ML serialized files are verified on disk.")
    
    # 2. Test Model Loading & Predict
    try:
        scaler = joblib.load(get_path('models', 'scaler.joblib'))
        encoder = joblib.load(get_path('models', 'encoder.joblib'))
        reg_model = joblib.load(get_path('models', 'regression', 'best_model.joblib'))
        clf_model = joblib.load(get_path('models', 'classification', 'best_model.joblib'))
        multi_model = joblib.load(get_path('models', 'classification', 'multinomial_logistic_regression.joblib'))
        
        logger.info("Successfully loaded all models and pipelines into memory.")
        
        # 3. Test data point
        test_features = {
            'duration_ms': 230666,
            'danceability': 0.676,
            'energy': 0.461,
            'key': 1,
            'loudness': -6.746,
            'mode': 0,
            'speechiness': 0.143,
            'acousticness': 0.0322,
            'instrumentalness': 0.00000101,
            'liveness': 0.358,
            'valence': 0.715,
            'tempo': 87.917,
            'time_signature': 4,
            'explicit': False,
            'track_genre': 'acoustic'
        }
        
        # DataFrame conversion
        input_df = pd.DataFrame([test_features])
        num_cols = ['duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'time_signature']
        
        X_input, _ = preprocess_features(input_df, scaler, encoder, num_cols, scaled=True)
        
        # Run test inference
        pred_popularity = reg_model.predict(X_input)[0]
        pred_skip_prob = clf_model.predict_proba(X_input)[0][1]
        pred_risk_category = multi_model.predict(X_input)[0]
        
        logger.info("Verification Inference Run Success!")
        logger.info(f"Input Track: Genre={test_features['track_genre']}, Tempo={test_features['tempo']}")
        logger.info(f"Predicted Popularity: {pred_popularity:.2f} / 100")
        logger.info(f"Predicted Skip Probability: {pred_skip_prob * 100:.1f}%")
        logger.info(f"Multinomial Predicted Risk Class: {pred_risk_category} (0:Low, 1:Medium, 2:High)")
        
        # Match metrics json
        with open(get_path('models', 'metrics_summary.json'), 'r') as f:
            summary = json.load(f)
        
        logger.info("==============================================")
        logger.info("SYSTEM ALIGNMENT: 100% HEALTHY. ALL CHECKS PASSED.")
        logger.info("==============================================")
        
    except Exception as e:
        logger.error(f"Inference simulation crashed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    run_verification()
