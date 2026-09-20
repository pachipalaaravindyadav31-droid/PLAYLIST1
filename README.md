# PlaylistPulse: AI-Based Music Popularity Prediction and Skip Behavior Analysis

PlaylistPulse is a comprehensive, production-grade academic Machine Learning and Web Dashboard project. Using a Spotify Tracks dataset containing 114,000 records, the project constructs a complete data pipeline to perform continuous popularity regression forecasting and binary/multiclass user track-skipping predictions.

## Project Structure

```
c:\playlist/
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependency file
├── README.md                  # Project documentation
├── data/
│   ├── dataset.csv            # Original dataset (already in workspace)
│   └── clean_sample.csv       # Preprocessed baseline subset for statistics
├── models/                    # Serialized models and pipelines
│   ├── scaler.joblib          # Fitted StandardScaler
│   ├── encoder.joblib         # Fitted OneHotEncoder for genre
│   ├── metrics_summary.json   # Pre-computed model comparison and tuning sweep metrics
│   ├── regression/
│   │   ├── best_model.joblib  # Selected best regression model (Linear Regression)
│   │   └── [model_name].joblib# Individual regression estimators
│   └── classification/
│       ├── best_model.joblib  # Selected best classification model (Logistic Regression)
│       ├── multinomial_logistic_regression.joblib # 3-class risk category classifier
│       └── [model_name].joblib# Individual classification estimators
├── src/                       # Python module package
│   ├── __init__.py
│   ├── utils.py               # Path management, loggers, and plot saving helpers
│   ├── data_preprocessing.py  # Cleans dataset, creates skip behavior proxies, fits pipelines
│   ├── train_regression.py    # Trains popularity estimators & runs tuning sweeps
│   ├── train_classification.py# Trains skip behavior estimators & multinomial logreg
│   ├── train_all.py           # Orchestrator script executing the entire train suite
│   └── verify.py              # Self-checks model deserialization and pipeline integrity
├── templates/                 # HTML UI layouts
│   ├── base.html              # Shell navbar & footer
│   ├── index.html             # Dashboard Overview & Comparison tables
│   ├── eda.html               # Audio feature distribution & correlation matrix views
│   ├── experiments.html       # Underfitting, Regularization paths, & scaling analyses
│   ├── predict.html           # Prediction form Playground & visual trace flowchart
│   └── monitoring.html        # Database log telemetry & Data Drift Detector
└── static/                    # Dashboard styles and visualizations
    ├── css/
    │   └── style.css          # Customized dark theme layout stylesheet
    ├── js/
    │   └── main.js            # Live range syncs & sequential flowchart animations
    └── images/                # Generated EDA and training sweep visual plots
```

---

## Installation & Setup

Ensure Python is installed. We recommend installing the package requirements into a clean environment:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Offline ML Training Pipeline** (Optional, as models and plots are pre-trained and saved):
   ```bash
   python -m src.train_all
   ```
   *This command cleans the dataset, fits and saves preprocessors, trains all 16 regression and classification models, runs tuning sweeps, produces the graphical visualizations, and compiles the comparison summary metrics into `models/metrics_summary.json`.*

3. **Verify Pipeline Alignment**:
   ```bash
   python -m src.verify
   ```
   *Verifies model loads, preprocessing matrices, and simulates a prediction query to output the system health status.*

4. **Launch Flask Web Server**:
   ```bash
   python app.py
   ```
   *Starts the server at `http://127.0.0.1:5000`.*

---

## Curriculum & Concept Mapping (M1, M2, M3)

This project explicitly implements and demonstrates the entire three-module Machine Learning curriculum:

### 🧩 M1. The ML System Lifecycle & Engineering
- **Production Prediction Service**: Running live on day one via Flask REST API (`POST /api/predict`) and interactive web dashboard.
- **Complete ML Lifecycle**: Raw CSV Data (`data/`) $\rightarrow$ Preprocessed dataset $\rightarrow$ Feature engineered proxies $\rightarrow$ Standardized train/test splits $\rightarrow$ Serialized Joblib models (`models/`) $\rightarrow$ Deployed HTTP endpoint $\rightarrow$ Monitored production log telemetry (`logs/predictions.csv`).
- **Code Mapping**: Clear modular separation across `src/data_preprocessing.py`, `src/train_regression.py`, `src/train_classification.py`, `app.py`, and `src/verify.py`.
- **Training-Serving Boundary**: Solves training-serving skew by storing fitted `StandardScaler` and `OneHotEncoder` pipelines as artifacts and loading identical transformers at inference time.
- **End-to-End Prediction Trace**: Visualized sequentially in the UI (`User Input` $\rightarrow$ `Flask API` $\rightarrow$ `Validation` $\rightarrow$ `Preprocessing` $\rightarrow$ `Saved Model` $\rightarrow$ `Log Telemetry` $\rightarrow$ `JSON Response`).
- **Data Drift Detection**: Dynamic sliding window comparisons of recent input features against training baseline means.

### 📐 M2. Supervised Learning — Linear Models at Depth
- **Linear Regression**: Ordinary Least Squares (OLS) baseline predicting track popularity.
- **Feature Scaling**: Demonstrates the mathematical necessity of `StandardScaler` ($z = \frac{x-\mu}{\sigma}$) for linear models. Shows scaling increases test $R^2$ from **0.036** to **0.264** and makes weights directly interpretable.
- **L1 / L2 / Elastic Net Regularization**:
  - **Ridge ($L_2$)**: Smooth coefficient shrinkage minimizing $\|y - Xw\|_2^2 + \lambda \|w\|_2^2$.
  - **Lasso ($L_1$)**: Rhombus constraint with sharp vertices driving non-essential coefficients to exactly zero (automatic feature selection).
  - **Elastic Net**: Blends both $L_1$ and $L_2$ penalties.
  - Interactive alpha sweeps and regularization path plots (`static/images/regression_regularization_path.png`).
- **Binary Logistic Regression**: Sigmoid activation $\sigma(w^T x)$ with binary cross-entropy loss, scoring **92.7% accuracy** and **0.973 ROC-AUC** for skip prediction.
- **Multinomial Logistic Regression**: Softmax cross-entropy classifier directly predicting 3-class skip risk categories (Low, Medium, High).
- **Missing Value Handling & Categorical Encoding**: Text imputation + One-Hot vectorization of 114 Spotify genres.

### 🌲 M3. Supervised Learning — Tree-Based Models & Ensembles
- **Decision Trees**: Greedy recursive splitting (MSE for regression, Gini for classification).
- **Bias-Variance Overfitting Curve**: Systematic sweep of tree `max_depth` from 1 to 15, empirically demonstrating high bias underfitting (depth 1-4) vs high variance overfitting (depth 8-15).
- **Random Forests**: Bagging (Bootstrap Aggregating) + random feature subsampling to reduce variance ($\text{Var} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$).
- **Boosting**: Gradient Boosting Regressor and Classifier performing functional gradient descent on residual errors.
- **Industrial Workhorses**: XGBoost (2nd order Taylor expansion + tree complexity regularization) and LightGBM (histogram-based bucketing + leaf-wise growth).
- **Feature Importance**: Gini / MDI importance bar plots extracted from ensembles, demonstrating why tree models dominate structured tabular datasets.

---

## Web API Endpoint

The Flask web app exposes a REST API for real-time predictions.

### `POST /api/predict`

* **Request Headers**: `Content-Type: application/json`
* **JSON Body**:
  ```json
  {
    "features": {
      "duration_ms": 230000,
      "danceability": 0.67,
      "energy": 0.46,
      "key": 1,
      "loudness": -6.74,
      "mode": 0,
      "speechiness": 0.14,
      "acousticness": 0.03,
      "instrumentalness": 0.00,
      "liveness": 0.36,
      "valence": 0.71,
      "tempo": 88.0,
      "time_signature": 4,
      "explicit": false,
      "track_genre": "acoustic"
    },
    "regression_model": "best_model",
    "classification_model": "best_model"
  }
  ```
  *(Note: You can pass `"best_model"` or specific model keys like `"linear_regression"`, `"random_forest"`, `"xgboost"`, etc.)*

* **JSON Response**:
  ```json
  {
    "model_version": "1.0.0",
    "models_used": {
      "classification": "logistic_regression",
      "regression": "linear_regression"
    },
    "popularity_prediction": 42.433291888497645,
    "skip_prediction_probability": 0.007963365691035252,
    "skip_risk_category": "Low"
  }
  ```

---

## Dynamic Deployment & Monitoring

- **Prediction Logging**: Every inference request sent through the `/api/predict` API is logged to `logs/predictions.csv` along with timestamps, inputs, prediction values, selected models, and the model version.
- **Data Drift Detection**: The dashboard calculates a sliding mean of the last 50 queries for features like `danceability`, `energy`, `acousticness`, and `speechiness`, and compares them against the training dataset baseline means. A drift warning flag is raised if the absolute difference exceeds $\Delta = 0.10$.
