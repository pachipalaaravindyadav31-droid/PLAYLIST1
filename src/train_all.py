import os
import json
from src.data_preprocessing import prepare_data_and_preprocessors
from src.train_regression import train_and_save_all_regression_models
from src.train_classification import train_and_save_all_classification_models
from src.utils import get_path, get_logger

logger = get_logger(__name__)

def train_everything():
    """Orchestrates the entire ML pipeline: preprocessing, regression, and classification."""
    logger.info("==================================================")
    logger.info("STARTING COMPLETE PLAYLISTPULSE ML PIPELINE RUN")
    logger.info("==================================================")
    
    # 1. Preprocess data
    data = prepare_data_and_preprocessors(sample_size=30000)
    
    # 2. Train regression models
    reg_results = train_and_save_all_regression_models(data)
    
    # 3. Train classification models
    clf_results = train_and_save_all_classification_models(data)
    
    # 4. Consolidate metrics and experiments
    summary = {
        'model_version': '1.0.0',
        'regression': reg_results,
        'classification': clf_results
    }
    
    # Save to models/metrics_summary.json
    summary_path = get_path('models', 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info("==================================================")
    logger.info(f"ML PIPELINE COMPLETED SUCCESSFULLY! Summary saved to {summary_path}")
    logger.info("==================================================")
    
    return summary

if __name__ == '__main__':
    train_everything()
