import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

from src.data_preprocessing import prepare_data_and_preprocessors
from src.utils import get_path, get_logger, save_plot

logger = get_logger(__name__)

def evaluate_binary_classifier(model, X_train, y_train, X_test, y_test, name):
    """Evaluates a binary classifier and returns a dict of metrics."""
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba (e.g. SVM if we had one)
        y_prob_train = y_pred_train
        y_prob_test = y_pred_test
        
    # Metrics
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    
    logger.info(f"Model: {name:25s} | Test Acc: {acc_test:6.3f}, F1: {f1_test:6.3f}, ROC-AUC: {roc_auc_test:6.3f}")
    
    return {
        'model_name': name,
        'train': {'accuracy': acc_train},
        'test': {
            'accuracy': acc_test,
            'precision': prec_test,
            'recall': rec_test,
            'f1': f1_test,
            'roc_auc': roc_auc_test
        },
        'y_prob_test': y_prob_test.tolist(),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }

def evaluate_multiclass_classifier(model, X_train, y_train, X_test, y_test, name):
    """Evaluates a multiclass classifier (e.g. Multinomial Logistic Regression) and returns metrics."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test) # shape: (N, 3)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    roc_auc_test = roc_auc_score(y_test, y_prob_test, average='macro', multi_class='ovr')
    
    logger.info(f"Multiclass Model: {name:25s} | Test Acc: {acc_test:6.3f}, Macro F1: {f1_test:6.3f}, Macro AUC: {roc_auc_test:6.3f}")
    
    return {
        'model_name': name,
        'test': {
            'accuracy': acc_test,
            'precision': prec_test,
            'recall': rec_test,
            'f1': f1_test,
            'roc_auc': roc_auc_test
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }

def train_and_save_all_classification_models(data):
    """Trains classification models, generates plots, and returns comparison dictionary."""
    logger.info("Starting Offline Classification Model Training...")
    
    # Data splits
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train_bin = data['y_train_clf_bin']
    y_test_bin = data['y_test_clf_bin']
    y_train_multi = data['y_train_clf_multi']
    y_test_multi = data['y_test_clf_multi']
    feature_names = data['feature_names']
    
    clf_dir = get_path('models', 'classification')
    os.makedirs(clf_dir, exist_ok=True)
    
    # 1. Binary Classification Models
    binary_models = {
        'logistic_regression': LogisticRegression(max_iter=2000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
        'xgboost': XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        'lightgbm': LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, verbosity=-1, n_jobs=-1)
    }
    
    binary_results = {}
    
    # Train and evaluate binary models
    for name, model in binary_models.items():
        logger.info(f"Training binary classifier: {name}...")
        model.fit(X_train, y_train_bin)
        
        # Save model
        joblib.dump(model, os.path.join(clf_dir, f'{name}.joblib'))
        
        # Evaluate
        metrics = evaluate_binary_classifier(model, X_train, y_train_bin, X_test, y_test_bin, name)
        binary_results[name] = metrics
        
    # Determine best binary model based on F1-score
    best_bin_name = max(binary_results.keys(), key=lambda k: binary_results[k]['test']['f1'])
    best_f1 = binary_results[best_bin_name]['test']['f1']
    logger.info(f"Best Binary Classification Model: {best_bin_name} with Test F1: {best_f1:.4f}")
    
    # Save best model separately
    joblib.dump(binary_models[best_bin_name], os.path.join(clf_dir, 'best_model.joblib'))
    
    # 2. Train Multinomial Logistic Regression (3-class risk predictor)
    logger.info("Training Multinomial Logistic Regression on 3-class Skip Risk Category...")
    multinomial_model = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)
    multinomial_model.fit(X_train, y_train_multi)
    
    # Save model
    joblib.dump(multinomial_model, os.path.join(clf_dir, 'multinomial_logistic_regression.joblib'))
    
    # Evaluate multinomial
    multi_results = evaluate_multiclass_classifier(
        multinomial_model, X_train, y_train_multi, X_test, y_test_multi, 'Multinomial Logistic Regression'
    )
    
    # 3. Generate ROC Curves for Binary Classifiers
    plt.figure(figsize=(8, 6))
    for name, res in binary_results.items():
        y_prob = res['y_prob_test']
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
        auc_val = res['test']['roc_auc']
        plt.plot(fpr, tpr, label=f"{res['model_name']} (AUC = {auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot(plt, 'classification_roc_curve.png')
    
    # 4. Generate Confusion Matrix for Best Binary Classifier
    plt.figure(figsize=(6, 5))
    cm_bin = np.array(binary_results[best_bin_name]['confusion_matrix'])
    sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Not Skipped (0)', 'Skipped (1)'], 
                yticklabels=['Not Skipped (0)', 'Skipped (1)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - Best Model ({binary_results[best_bin_name]["model_name"]})', fontsize=13)
    save_plot(plt, 'classification_confusion_matrix.png')
    
    # 5. Generate Confusion Matrix for Multinomial Logistic Regression
    plt.figure(figsize=(7, 6))
    cm_multi = np.array(multi_results['confusion_matrix'])
    sns.heatmap(cm_multi, annot=True, fmt="d", cmap="Purples",
                xticklabels=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)'],
                yticklabels=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Multinomial Logistic Regression (3-Class)', fontsize=13)
    save_plot(plt, 'classification_multinomial_cm.png')
    
    # 6. Feature Importance for Tree Classification Model (using RandomForestClassifier)
    best_clf_tree = 'random_forest'
    if best_clf_tree in binary_models:
        rf = binary_models[best_clf_tree]
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:15]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, palette='magma')
        plt.title('Top 15 Feature Importances for Skip Behavior (Random Forest Classifier)', fontsize=14)
        plt.xlabel('Importance Value', fontsize=12)
        save_plot(plt, 'classification_feature_importance.png')
        logger.info("Classification Feature importance plot saved.")
        
    # Clean temporary probability vectors from results to keep metadata JSON small
    for name in binary_results:
        if 'y_prob_test' in binary_results[name]:
            del binary_results[name]['y_prob_test']
            
    return {
        'binary_comparison': binary_results,
        'best_binary_model': best_bin_name,
        'multinomial_results': multi_results
    }

if __name__ == '__main__':
    data = prepare_data_and_preprocessors(sample_size=30000)
    results = train_and_save_all_classification_models(data)
    print("Done Training Classification Models. Best Binary Model:", results['best_binary_model'])
