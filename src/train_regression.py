import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import prepare_data_and_preprocessors
from src.utils import get_path, get_logger, save_plot

logger = get_logger(__name__)

def evaluate_regressor(model, X_train, y_train, X_test, y_test, name):
    """Evaluate a regressor and return a dict of metrics."""
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Train Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Test Metrics
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    logger.info(f"Model: {name:25s} | Train R2: {r2_train:6.3f}, Test R2: {r2_test:6.3f} | Test RMSE: {rmse_test:6.3f}")
    
    return {
        'model_name': name,
        'train': {'mae': mae_train, 'mse': mse_train, 'rmse': rmse_train, 'r2': r2_train},
        'test': {'mae': mae_test, 'mse': mse_test, 'rmse': rmse_test, 'r2': r2_test}
    }

def run_scaling_experiment(data):
    """Runs Linear Regression with and without scaled features and returns metrics."""
    logger.info("Running Feature Scaling Experiment...")
    
    X_train_unscaled = data['X_train_unscaled']
    X_test_unscaled = data['X_test_unscaled']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train_reg']
    y_test = data['y_test_reg']
    
    # Train unscaled
    lr_unscaled = LinearRegression()
    lr_unscaled.fit(X_train_unscaled, y_train)
    m_unscaled = evaluate_regressor(lr_unscaled, X_train_unscaled, y_train, X_test_unscaled, y_test, 'Linear Regression (Unscaled)')
    
    # Train scaled
    lr_scaled = LinearRegression()
    lr_scaled.fit(X_train_scaled, y_train)
    m_scaled = evaluate_regressor(lr_scaled, X_train_scaled, y_train, X_test_scaled, y_test, 'Linear Regression (Scaled)')
    
    # Let's select the first 13 coefficients (which correspond to the original audio features)
    num_cols = data['num_cols']
    coef_unscaled = lr_unscaled.coef_[:13]
    coef_scaled = lr_scaled.coef_[:13]
    
    # Save a comparison bar plot of the coefficients
    plt.figure(figsize=(12, 6))
    x = np.arange(len(num_cols))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.barh(x - width/2, coef_unscaled, width, label='Unscaled', color='gray')
    plt.yticks(x, num_cols)
    plt.title('Unscaled Coefficients (Arbitrary Scales)')
    plt.xlabel('Coefficient Value')
    
    plt.subplot(1, 2, 2)
    plt.barh(x + width/2, coef_scaled, width, label='Scaled', color='royalblue')
    plt.yticks(x, num_cols)
    plt.title('Scaled Coefficients (Directly Comparable)')
    plt.xlabel('Coefficient Value')
    
    plt.suptitle('Effect of Feature Scaling on Linear Regression Coefficients', fontsize=14)
    plt.tight_layout()
    save_plot(plt, 'regression_scaling_coefs.png')
    
    return {
        'unscaled_metrics': m_unscaled,
        'scaled_metrics': m_scaled
    }

def run_regularization_experiment(data):
    """Runs Ridge, Lasso, and Elastic Net over multiple alphas and plots results."""
    logger.info("Running L1/L2 Regularization Alpha Sweep...")
    
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train_reg']
    y_test = data['y_test_reg']
    num_cols = data['num_cols']
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    
    ridge_mses = []
    lasso_mses = []
    enet_mses = []
    
    # Track coefficients for paths (using the first 13 audio features)
    ridge_coefs = []
    lasso_coefs = []
    
    for a in alphas:
        # Ridge
        ridge = Ridge(alpha=a, random_state=42)
        ridge.fit(X_train, y_train)
        ridge_mses.append(mean_squared_error(y_test, ridge.predict(X_test)))
        ridge_coefs.append(ridge.coef_[:13])
        
        # Lasso
        lasso = Lasso(alpha=a, random_state=42, max_iter=2000)
        lasso.fit(X_train, y_train)
        lasso_mses.append(mean_squared_error(y_test, lasso.predict(X_test)))
        lasso_coefs.append(lasso.coef_[:13])
        
        # Elastic Net (l1_ratio=0.5)
        enet = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42, max_iter=2000)
        enet.fit(X_train, y_train)
        enet_mses.append(mean_squared_error(y_test, enet.predict(X_test)))
        
    # Plot Alpha vs MSE
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, ridge_mses, marker='o', label='Ridge (L2)', color='blue')
    plt.plot(alphas, lasso_mses, marker='s', label='Lasso (L1)', color='red')
    plt.plot(alphas, enet_mses, marker='^', label='Elastic Net (L1+L2)', color='purple')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Test Mean Squared Error (MSE)')
    plt.title('Regularization Tuning: Alpha vs Test MSE')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    save_plot(plt, 'regression_alpha_comparison.png')
    
    # Plot Coefficient Shrinkage Paths
    plt.figure(figsize=(12, 5))
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    plt.subplot(1, 2, 1)
    for i, col in enumerate(num_cols):
        plt.plot(alphas, ridge_coefs[:, i], marker='.', label=col)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge (L2) Coefficient Shrinkage Path')
    
    plt.subplot(1, 2, 2)
    for i, col in enumerate(num_cols):
        plt.plot(alphas, lasso_coefs[:, i], marker='.', label=col)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso (L1) Coefficient Shrinkage Path')
    
    plt.suptitle('Weight Shrinkage (Regularization Paths)', fontsize=14)
    plt.tight_layout()
    save_plot(plt, 'regression_regularization_path.png')
    
    return {
        'alphas': alphas,
        'ridge_mses': ridge_mses,
        'lasso_mses': lasso_mses,
        'enet_mses': enet_mses
    }

def run_decision_tree_experiment(data):
    """Varies DT depth and plots training vs validation error to show bias-variance trade-off."""
    logger.info("Running Decision Tree Depth Experiment...")
    
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train_reg']
    y_test = data['y_test_reg']
    
    depths = list(range(1, 16))
    train_mses = []
    test_mses = []
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        train_mses.append(mean_squared_error(y_train, dt.predict(X_train)))
        test_mses.append(mean_squared_error(y_test, dt.predict(X_test)))
        
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_mses, marker='o', label='Training Error (Bias)', color='blue')
    plt.plot(depths, test_mses, marker='s', label='Testing Error (Variance)', color='red')
    plt.xlabel('Decision Tree Max Depth')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Decision Tree Overfitting: Max Depth vs Train/Test MSE')
    
    # Highlight underfitting and overfitting regions
    plt.axvspan(1, 4, alpha=0.15, color='gray', label='Underfitting Zone (High Bias)')
    plt.axvspan(8, 15, alpha=0.15, color='orange', label='Overfitting Zone (High Variance)')
    
    plt.legend()
    plt.xticks(depths)
    plt.grid(True)
    save_plot(plt, 'regression_dt_depth.png')
    
    return {
        'depths': depths,
        'train_mses': train_mses,
        'test_mses': test_mses
    }

def train_and_save_all_regression_models(data):
    """Trains the 9 regression models, runs experiments, saves models, and generates metrics."""
    logger.info("Starting Offline Regression Model Training...")
    
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train_reg']
    y_test = data['y_test_reg']
    feature_names = data['feature_names']
    
    # 1. Initialize models
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
        'decision_tree': DecisionTreeRegressor(max_depth=6, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        'xgboost': XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        'lightgbm': LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, verbosity=-1, n_jobs=-1)
    }
    
    results = {}
    reg_dir = get_path('models', 'regression')
    os.makedirs(reg_dir, exist_ok=True)
    
    # 2. Train and evaluate
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, os.path.join(reg_dir, f'{name}.joblib'))
        
        # Evaluate
        metrics = evaluate_regressor(model, X_train, y_train, X_test, y_test, name)
        results[name] = metrics
        
    # 3. Find and save the best model separately
    best_model_name = max(results.keys(), key=lambda k: results[k]['test']['r2'])
    best_r2 = results[best_model_name]['test']['r2']
    logger.info(f"Best Regression Model: {best_model_name} with Test R2: {best_r2:.4f}")
    
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join(reg_dir, 'best_model.joblib'))
    
    # 4. Feature Importance for Tree Models (using best tree model or RandomForest)
    best_tree_name = 'random_forest'
    if best_tree_name in models:
        rf = models[best_tree_name]
        importances = rf.feature_importances_
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Select top 15 features
        top_indices = indices[:15]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, palette='viridis')
        plt.title('Top 15 Feature Importances for Spotify popularity (Random Forest)', fontsize=14)
        plt.xlabel('Importance Value', fontsize=12)
        save_plot(plt, 'regression_feature_importance.png')
        logger.info("Feature importance plot saved.")
        
    # 5. Run Experiments
    scaling_exp = run_scaling_experiment(data)
    reg_exp = run_regularization_experiment(data)
    dt_exp = run_decision_tree_experiment(data)
    
    return {
        'comparison': results,
        'best_model': best_model_name,
        'experiments': {
            'scaling': scaling_exp,
            'regularization': {
                'alphas': reg_exp['alphas'],
                'ridge_mses': reg_exp['ridge_mses'],
                'lasso_mses': reg_exp['lasso_mses'],
                'enet_mses': reg_exp['enet_mses']
            },
            'decision_tree_depth': {
                'depths': dt_exp['depths'],
                'train_mses': dt_exp['train_mses'],
                'test_mses': dt_exp['test_mses']
            }
        }
    }

if __name__ == '__main__':
    data = prepare_data_and_preprocessors(sample_size=30000)
    results = train_and_save_all_regression_models(data)
    print("Done Training Regression Models. Best model:", results['best_model'])
