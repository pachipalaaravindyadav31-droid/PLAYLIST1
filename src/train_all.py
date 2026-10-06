"""
PlaylistPulse Master ML Pipeline Orchestrator (Modules M1, M2, M3, M4, M5)
Executes:
- Data Preprocessing & Proxy Synthesis (M1)
- Linear & Tree Regression Models for Popularity (M2 & M3)
- Linear & Tree Classification Models for Skip Behavior (M2 & M3)
- Unsupervised Learning: K-Means, PCA, DBSCAN, Isolation Forest (M4)
- Model Evaluation, Selection, Calibration, and Significance Testing (M5)
"""
import os
import json
from src.data_preprocessing import prepare_data_and_preprocessors
from src.train_regression import train_and_save_all_regression_models
from src.train_classification import train_and_save_all_classification_models
from src.train_unsupervised import train_and_evaluate_unsupervised
from src.evaluation_calibration import evaluate_and_calibrate_models
from src.utils import get_path, get_logger

logger = get_logger(__name__)


def train_everything(sample_size=30000, filepath=None):
    """Orchestrates the entire ML curriculum pipeline (M1-M5)."""
    logger.info("==================================================")
    logger.info("STARTING COMPLETE PLAYLISTPULSE ML PIPELINE (M1 - M5)")
    logger.info("==================================================")
    
    # 1. Preprocess data (M1)
    data = prepare_data_and_preprocessors(sample_size=sample_size, filepath=filepath)

    
    # 2. Train regression models (M2 & M3)
    reg_results = train_and_save_all_regression_models(data)
    
    # 3. Train classification models (M2 & M3)
    clf_results = train_and_save_all_classification_models(data)

    # 4. Train Unsupervised Learning Models (M4)
    unsup_results = train_and_evaluate_unsupervised(data, sample_size=min(5000, sample_size))

    # 5. Evaluate, Calibrate & Run Significance Tests (M5)
    calib_results = evaluate_and_calibrate_models(data)
    
    # 6. Consolidate metrics and experiments
    summary = {
        'model_version': '2.0.0',
        'regression': reg_results,
        'classification': clf_results,
        'unsupervised_m4': unsup_results,
        'evaluation_calibration_m5': calib_results
    }
    
    # Save to models/metrics_summary.json
    summary_path = get_path('models', 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    logger.info("==================================================")
    logger.info(f"ALL MODULES M1-M5 COMPLETED! Master Summary saved to {summary_path}")
    logger.info("==================================================")
    
    return summary


if __name__ == '__main__':
    train_everything()
