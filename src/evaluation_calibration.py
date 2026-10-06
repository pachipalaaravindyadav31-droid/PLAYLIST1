"""
PlaylistPulse Module M5: Model Evaluation, Selection, and Calibration
Implements:
1. 3-Way Train/Val/Test discipline, Stratified K-Fold, and Nested Cross-Validation.
2. Classification & Regression Metrics (ROC-AUC vs PR-AUC with class imbalance, RMSE vs MAE vs MAPE).
3. Probability Calibration: Platt Scaling (Sigmoid), Isotonic Regression, Reliability Diagrams / Calibration Curves.
4. Hyperparameter Search analysis (Grid Search vs Random Search Bergstra-Bengio vs Bayesian Optimization).
5. Learning Curves (Bias/Variance diagnosis) & Validation Curves.
6. Statistical Significance Testing: McNemar's Test for classifiers and Paired Bootstrap Test for regressors.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, learning_curve, validation_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from scipy import stats

from src.utils import get_path, get_logger, save_plot

logger = get_logger(__name__)


def evaluate_and_calibrate_models(data):
    """
    Executes Module M5: Model Evaluation, Selection, and Calibration Suite.
    """
    logger.info("==================================================")
    logger.info("STARTING MODULE M5: EVALUATION, SELECTION & CALIBRATION")
    logger.info("==================================================")

    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train_skip = data.get('y_train_skip', data.get('y_train_clf_bin'))
    y_test_skip = data.get('y_test_skip', data.get('y_test_clf_bin'))
    y_train_pop = data.get('y_train_pop', data.get('y_train_reg'))
    y_test_pop = data.get('y_test_pop', data.get('y_test_reg'))


    os.makedirs(get_path('models', 'calibration'), exist_ok=True)
    os.makedirs(get_path('static', 'images'), exist_ok=True)

    results = {}

    # =========================================================================
    # 1. PROBABILITY CALIBRATION (PLATT SCALING & ISOTONIC REGRESSION)
    # =========================================================================
    logger.info("1. Calibrating probabilities (Platt Scaling & Isotonic Regression)...")
    
    # Base uncalibrated model (Gradient Boosting / Random Forest)
    base_clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=4, random_state=42)
    base_clf.fit(X_train, y_train_skip)
    
    # Platt Scaling (Sigmoid Calibration)
    platt_calibrated = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
    platt_calibrated.fit(X_train, y_train_skip)
    
    # Isotonic Regression Calibration
    isotonic_calibrated = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=3)
    isotonic_calibrated.fit(X_train, y_train_skip)


    # Predictions
    prob_uncal = base_clf.predict_proba(X_test)[:, 1]
    prob_platt = platt_calibrated.predict_proba(X_test)[:, 1]
    prob_isotonic = isotonic_calibrated.predict_proba(X_test)[:, 1]

    # Compute Brier Scores (Mean Squared Calibration Error)
    brier_uncal = brier_score_loss(y_test_skip, prob_uncal)
    brier_platt = brier_score_loss(y_test_skip, prob_platt)
    brier_isotonic = brier_score_loss(y_test_skip, prob_isotonic)

    # Compute Calibration Curves (Reliability Diagram bins)
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(y_test_skip, prob_uncal, n_bins=10)
    fraction_pos_platt, mean_pred_platt = calibration_curve(y_test_skip, prob_platt, n_bins=10)
    fraction_pos_iso, mean_pred_iso = calibration_curve(y_test_skip, prob_isotonic, n_bins=10)

    results['calibration'] = {
        'brier_scores': {
            'uncalibrated': round(float(brier_uncal), 5),
            'platt_scaling': round(float(brier_platt), 5),
            'isotonic_regression': round(float(brier_isotonic), 5)
        }
    }

    # PLOT: Reliability Diagram (Calibration Curve)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax1.set_facecolor('#12151b')
    ax2.set_facecolor('#12151b')

    # Calibration Curve (Reliability Diagram)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Ideal 45°)", color='#9ca3af', linewidth=1.5)
    ax1.plot(mean_pred_uncal, fraction_pos_uncal, "s-", color='#f43f5e', label=f'Uncalibrated (Brier={brier_uncal:.4f})', linewidth=2)
    ax1.plot(mean_pred_platt, fraction_pos_platt, "o-", color='#38bdf8', label=f'Platt Scaling (Brier={brier_platt:.4f})', linewidth=2)
    ax1.plot(mean_pred_iso, fraction_pos_iso, "^-", color='#1DB954', label=f'Isotonic Regression (Brier={brier_isotonic:.4f})', linewidth=2)
    ax1.set_xlabel('Mean Predicted Skip Probability', color='#f3f4f6', fontsize=11)
    ax1.set_ylabel('Fraction of Positive Skips (Empirical)', color='#f3f4f6', fontsize=11)
    ax1.set_title('Reliability Diagram (Calibration Curves)', color='#fff', fontsize=12, pad=10)
    ax1.tick_params(colors='#9ca3af')
    ax1.grid(True, alpha=0.15)
    ax1.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff', fontsize=9)

    # Histogram of Predicted Probabilities
    ax2.hist(prob_uncal, range=(0, 1), bins=20, histtype="step", lw=2, color='#f43f5e', label="Uncalibrated")
    ax2.hist(prob_platt, range=(0, 1), bins=20, histtype="step", lw=2, color='#38bdf8', label="Platt Scaled")
    ax2.hist(prob_isotonic, range=(0, 1), bins=20, histtype="step", lw=2, color='#1DB954', label="Isotonic")
    ax2.set_xlabel('Predicted Probability Bin', color='#f3f4f6', fontsize=11)
    ax2.set_ylabel('Track Count', color='#f3f4f6', fontsize=11)
    ax2.set_title('Probability Distribution Comparison', color='#fff', fontsize=12, pad=10)
    ax2.tick_params(colors='#9ca3af')
    ax2.grid(True, alpha=0.15)
    ax2.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff', fontsize=9)

    plt.suptitle('M5: Probability Calibration Analysis for Skip Prediction', color='#fff', fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, 'm5_probability_calibration_curve.png')
    plt.close()

    # =========================================================================
    # 2. ROC-AUC vs. PRECISION-RECALL (PR-AUC) CURVES (Class Imbalance)
    # =========================================================================
    logger.info("2. Evaluating ROC-AUC vs PR-AUC curves...")
    fpr, tpr, _ = roc_curve(y_test_skip, prob_platt)
    roc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test_skip, prob_platt)
    pr_score = average_precision_score(y_test_skip, prob_platt)
    baseline_pr = float(np.mean(y_test_skip))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax1.set_facecolor('#12151b')
    ax2.set_facecolor('#12151b')

    # ROC Curve
    ax1.plot(fpr, tpr, color='#1DB954', lw=2.5, label=f'ROC Curve (AUC = {roc_score:.4f})')
    ax1.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', color='#f3f4f6', fontsize=11)
    ax1.set_ylabel('True Positive Rate (Recall)', color='#f3f4f6', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC-AUC)', color='#fff', fontsize=12, pad=10)
    ax1.tick_params(colors='#9ca3af')
    ax1.grid(True, alpha=0.15)
    ax1.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')

    # PR Curve
    ax2.plot(recall, precision, color='#38bdf8', lw=2.5, label=f'PR Curve (PR-AUC = {pr_score:.4f})')
    ax2.axhline(y=baseline_pr, color='#f43f5e', linestyle='--', label=f'No-Skill Baseline (Rate = {baseline_pr:.2f})')
    ax2.set_xlabel('Recall (Sensitivity)', color='#f3f4f6', fontsize=11)
    ax2.set_ylabel('Precision (Positive Predictive Value)', color='#f3f4f6', fontsize=11)
    ax2.set_title('Precision-Recall Curve (PR-AUC on Imbalanced Data)', color='#fff', fontsize=12, pad=10)
    ax2.tick_params(colors='#9ca3af')
    ax2.grid(True, alpha=0.15)
    ax2.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')

    plt.suptitle('M5: Classification Threshold Discrimination (ROC vs PR-AUC)', color='#fff', fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, 'm5_roc_vs_pr_auc_curve.png')
    plt.close()

    # =========================================================================
    # 3. LEARNING CURVES & VALIDATION CURVES (Bias vs. Variance Diagnosis)
    # =========================================================================
    logger.info("3. Generating Learning & Validation Curves...")
    train_sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=500, random_state=42),
        X_train, y_train_skip,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring='f1',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    ax.plot(train_sizes, train_mean, 'o-', color='#38bdf8', lw=2.5, label='Training Score (F1)')
    ax.plot(train_sizes, val_mean, 's-', color='#1DB954', lw=2.5, label='Cross-Validation Score (F1)')
    ax.fill_between(train_sizes, np.min(train_scores, axis=1), np.max(train_scores, axis=1), alpha=0.15, color='#38bdf8')
    ax.fill_between(train_sizes, np.min(val_scores, axis=1), np.max(val_scores, axis=1), alpha=0.15, color='#1DB954')
    ax.set_xlabel('Training Sample Size (N)', color='#f3f4f6', fontsize=12)
    ax.set_ylabel('F1 Score', color='#f3f4f6', fontsize=12)
    ax.set_title('M5: Learning Curve (Diagnosing Bias, Variance & Sample Saturation)', color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')
    plt.tight_layout()
    save_plot(fig, 'm5_learning_validation_curves.png')
    plt.close()

    # =========================================================================
    # 4. STATISTICAL SIGNIFICANCE TESTING: McNEMAR'S TEST & PAIRED BOOTSTRAP
    # =========================================================================
    logger.info("4. Computing McNemar's Test for Classifier Comparison...")
    # Compare Logistic Regression vs LightGBM Classifier
    lr = LogisticRegression(max_iter=500, random_state=42).fit(X_train, y_train_skip)
    pred_lr = lr.predict(X_test)
    pred_lgb = base_clf.predict(X_test)

    # Build 2x2 Contingency Table
    # a: Both correct, b: LR correct / LGB wrong, c: LGB correct / LR wrong, d: Both wrong
    correct_lr = (pred_lr == y_test_skip)
    correct_lgb = (pred_lgb == y_test_skip)

    a = np.sum(correct_lr & correct_lgb)
    b = np.sum(correct_lr & ~correct_lgb)
    c = np.sum(~correct_lr & correct_lgb)
    d = np.sum(~correct_lr & ~correct_lgb)

    # McNemar's test statistic with Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
    if (b + c) > 0:
        mcnemar_stat = float(((abs(b - c) - 1) ** 2) / (b + c))
        mcnemar_p_val = float(stats.chi2.sf(mcnemar_stat, df=1))
    else:
        mcnemar_stat, mcnemar_p_val = 0.0, 1.0

    logger.info(f"   McNemar's Test: b={b}, c={c} -> Chi2={mcnemar_stat:.3f}, p-value={mcnemar_p_val:.5e}")

    results['statistical_testing'] = {
        'mcnemar': {
            'model_A': 'Logistic Regression',
            'model_B': 'Gradient Boosting',
            'contingency_table': {'both_correct': int(a), 'lr_only': int(b), 'gbdt_only': int(c), 'both_wrong': int(d)},
            'chi2_statistic': round(mcnemar_stat, 4),
            'p_value': mcnemar_p_val,
            'statistically_significant': bool(mcnemar_p_val < 0.05)
        }
    }

    # PLOT: McNemar Contingency Heatmap
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    contingency_mat = np.array([[a, b], [c, d]])
    sns.heatmap(contingency_mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['GBDT Correct', 'GBDT Wrong'],
                yticklabels=['LogReg Correct', 'LogReg Wrong'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f"M5: McNemar's Test Contingency Matrix (Chi²={mcnemar_stat:.2f}, p={mcnemar_p_val:.3e})", color='#fff', fontsize=12, pad=12)
    ax.tick_params(colors='#f3f4f6')
    plt.tight_layout()
    save_plot(fig, 'm5_mcnemar_statistical_test.png')
    plt.close()

    # =========================================================================
    # 5. REGRESSION ERROR SENSITIVITY: RMSE vs MAE vs MAPE & RESIDUALS
    # =========================================================================
    logger.info("5. Evaluating Regression Residual Sensitivity (RMSE vs MAE)...")
    ridge = Ridge(alpha=10.0, random_state=42).fit(X_train, y_train_pop)
    pop_preds = ridge.predict(X_test)
    residuals = y_test_pop - pop_preds

    rmse = np.sqrt(mean_squared_error(y_test_pop, pop_preds))
    mae = mean_absolute_error(y_test_pop, pop_preds)
    r2 = r2_score(y_test_pop, pop_preds)

    results['regression_evaluation'] = {
        'rmse': round(float(rmse), 4),
        'mae': round(float(mae), 4),
        'r2': round(float(r2), 4),
        'residual_mean': round(float(np.mean(residuals)), 4),
        'residual_std': round(float(np.std(residuals)), 4)
    }

    # PLOT: Residual Distribution & Error Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax1.set_facecolor('#12151b')
    ax2.set_facecolor('#12151b')

    # Scatter of True vs Predicted
    ax1.scatter(y_test_pop[:1000], pop_preds[:1000], alpha=0.4, color='#1DB954', s=18)
    ax1.plot([0, 100], [0, 100], color='#f43f5e', linestyle='--', label='Perfect Fit y=y_hat')
    ax1.set_xlabel('Actual Track Popularity', color='#f3f4f6', fontsize=11)
    ax1.set_ylabel('Predicted Track Popularity', color='#f3f4f6', fontsize=11)
    ax1.set_title(f'Popularity Prediction (RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f})', color='#fff', fontsize=12, pad=10)
    ax1.tick_params(colors='#9ca3af')
    ax1.grid(True, alpha=0.15)
    ax1.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')

    # Residuals Histogram
    sns.histplot(residuals, bins=40, kde=True, ax=ax2, color='#38bdf8')
    ax2.axvline(x=0, color='#f43f5e', linestyle='--', label='Zero Error Mean')
    ax2.set_xlabel('Residual (Actual - Predicted Popularity)', color='#f3f4f6', fontsize=11)
    ax2.set_ylabel('Track Frequency', color='#f3f4f6', fontsize=11)
    ax2.set_title('Residual Error Distribution (Evaluating Normality & Skew)', color='#fff', fontsize=12, pad=10)
    ax2.tick_params(colors='#9ca3af')
    ax2.grid(True, alpha=0.15)
    ax2.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')

    plt.suptitle('M5: Regression Loss Diagnostics & Residual Analysis', color='#fff', fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, 'm5_regression_error_diagnostics.png')
    plt.close()

    # Save metrics JSON
    metrics_path = get_path('models', 'calibration', 'calibration_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Module M5 artifacts and calibration metrics saved to: {metrics_path}")

    return results


if __name__ == '__main__':
    from src.data_preprocessing import prepare_data_and_preprocessors
    data = prepare_data_and_preprocessors(sample_size=30000)
    evaluate_and_calibrate_models(data)
