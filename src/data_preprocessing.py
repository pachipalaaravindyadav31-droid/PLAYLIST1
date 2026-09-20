import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import get_logger, get_path, save_plot

logger = get_logger(__name__)

def load_and_clean_data(filepath):
    """Loads dataset from CSV, cleans missing values, and drops duplicates/unnecessary columns."""
    logger.info(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Clean missing values in text columns
    text_cols = ['artists', 'album_name', 'track_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    logger.info(f"Raw data loaded with shape: {df.shape}")
    return df

def generate_skip_proxies(df, seed=42):
    """Generates synthetic skip behavior features based on track properties plus noise."""
    logger.info("Generating synthetic skip behavior proxies...")
    np.random.seed(seed)
    
    # Formula for skip score (probability)
    # Higher acousticness, speechiness, liveness, and lower energy, danceability, popularity increase skip rate.
    dance = df['danceability']
    energy = df['energy']
    acoustic = df['acousticness']
    liveness = df['liveness']
    speech = df['speechiness']
    pop = df['popularity'] / 100.0
    
    skip_score = 0.3 * (1.0 - dance) + 0.2 * (1.0 - energy) + 0.15 * acoustic + 0.1 * liveness + 0.15 * speech + 0.1 * (1.0 - pop)
    
    # Add random noise to simulate human behavior stochasticity (prevents 100% accuracy)
    noise = np.random.normal(0, 0.05, size=len(df))
    skip_score = np.clip(skip_score + noise, 0.0, 1.0)
    
    df['skip_probability'] = skip_score
    
    # Binary skip target
    df['skipped'] = (skip_score > 0.45).astype(int)
    
    # 3-class skip risk category target (0: Low, 1: Medium, 2: High)
    df['skip_risk_category'] = df['skip_probability'].apply(
        lambda x: 0 if x < 0.35 else (1 if x < 0.60 else 2)
    )
    
    logger.info(f"Generated skip proxies. Value counts for 'skipped':\n{df['skipped'].value_counts(normalize=True)}")
    logger.info(f"Value counts for 'skip_risk_category':\n{df['skip_risk_category'].value_counts(normalize=True)}")
    return df

def fit_save_preprocessors(df_train, num_cols, cat_cols):
    """Fits and saves StandardScaler and OneHotEncoder on training data."""
    logger.info("Fitting preprocessors on training data...")
    models_dir = get_path('models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Standard Scaler for numeric columns
    scaler = StandardScaler()
    scaler.fit(df_train[num_cols])
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    logger.info("Scaler fitted and saved.")
    
    # One Hot Encoder for genre
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_train[['track_genre']])
    joblib.dump(encoder, os.path.join(models_dir, 'encoder.joblib'))
    logger.info("One-hot encoder fitted and saved.")
    
    return scaler, encoder

def preprocess_features(df, scaler, encoder, num_cols, scaled=True):
    """Transforms dataframe columns into a feature matrix X."""
    # Scale or copy numeric columns
    if scaled:
        X_num = scaler.transform(df[num_cols])
    else:
        X_num = df[num_cols].values
        
    # Explicit column (boolean -> float)
    X_exp = df[['explicit']].astype(float).values
    
    # One-hot encoded genre columns
    X_genre = encoder.transform(df[['track_genre']])
    
    # Combine features
    X = np.hstack([X_num, X_exp, X_genre])
    
    # Get feature names
    genre_feature_names = encoder.get_feature_names_out(['track_genre']).tolist()
    feature_names = num_cols + ['explicit'] + genre_feature_names
    
    return X, feature_names

def generate_eda_plots(df):
    """Generates and saves exploratory data analysis plots in static/images."""
    logger.info("Generating EDA plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Popularity Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['popularity'], kde=True, bins=30, color='royalblue')
    plt.title('Distribution of Track Popularity', fontsize=14)
    plt.xlabel('Popularity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    save_plot(plt, 'popularity_dist.png')
    
    # 2. Skip Score/Probability Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['skip_probability'], kde=True, bins=30, color='tomato')
    plt.axvline(0.35, color='green', linestyle='--', label='Low/Medium Risk Boundary')
    plt.axvline(0.60, color='red', linestyle='--', label='Medium/High Risk Boundary')
    plt.title('Distribution of Skip Probability Proxy', fontsize=14)
    plt.xlabel('Skip Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    save_plot(plt, 'skip_prob_dist.png')
    
    # 3. Correlation Heatmap (numerical features)
    num_cols_for_corr = ['popularity', 'duration_ms', 'danceability', 'energy', 
                         'loudness', 'speechiness', 'acousticness', 
                         'instrumentalness', 'liveness', 'valence', 'tempo']
    plt.figure(figsize=(10, 8))
    corr = df[num_cols_for_corr].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix of Audio Features & Popularity', fontsize=14)
    save_plot(plt, 'correlation_matrix.png')
    
    # 4. Danceability vs Energy vs Skip Probability
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['danceability'].sample(5000, random_state=42), 
                          df['energy'].sample(5000, random_state=42), 
                          c=df['skip_probability'].sample(5000, random_state=42), 
                          cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Skip Probability')
    plt.title('Danceability vs Energy Colored by Skip Probability (Sample of 5,000 Tracks)', fontsize=13)
    plt.xlabel('Danceability', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    save_plot(plt, 'dance_energy_skip.png')
    
    # 5. Skip Risk Category distribution
    plt.figure(figsize=(6, 5))
    risk_counts = df['skip_risk_category'].value_counts().sort_index()
    risk_labels = ['Low (0)', 'Medium (1)', 'High (2)']
    sns.barplot(x=risk_labels, y=risk_counts.values, palette='RdYlGn_r')
    plt.title('Distribution of Skip-Risk Categories', fontsize=14)
    plt.ylabel('Number of Tracks', fontsize=12)
    save_plot(plt, 'skip_risk_dist.png')
    
    logger.info("EDA plots saved to static/images.")

def prepare_data_and_preprocessors(sample_size=30000, test_size=0.2, seed=42):
    """Main data processing pipeline. Prepares raw data, creates splits, fits and saves preprocessors."""
    # 1. Load and clean
    csv_path = get_path('Data', 'dataset.csv')
    df = load_and_clean_data(csv_path)
    
    # 2. Generate target proxies
    df = generate_skip_proxies(df, seed=seed)
    
    # 3. Generate EDA plots on the full dataset
    generate_eda_plots(df)
    
    # 4. Save a small sample of clean data for UI / EDA overview stats
    clean_data_path = get_path('data', 'clean_sample.csv')
    os.makedirs(os.path.dirname(clean_data_path), exist_ok=True)
    df.sample(n=min(5000, len(df)), random_state=seed).to_csv(clean_data_path, index=False)
    
    # 5. Split train-test (on a downsampled representative portion for fast training)
    if sample_size and sample_size < len(df):
        logger.info(f"Downsampling training data to {sample_size} records...")
        df_sample = df.sample(n=sample_size, random_state=seed)
    else:
        df_sample = df
        
    df_train, df_test = train_test_split(df_sample, test_size=test_size, random_state=seed, stratify=df_sample['skipped'])
    logger.info(f"Split data: Train={df_train.shape[0]}, Test={df_test.shape[0]}")
    
    # Define columns
    num_cols = ['duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                'valence', 'tempo', 'time_signature']
    cat_cols = ['explicit']
    
    # 6. Fit and save preprocessors
    scaler, encoder = fit_save_preprocessors(df_train, num_cols, cat_cols)
    
    # 7. Generate features matrices
    X_train_scaled, feature_names = preprocess_features(df_train, scaler, encoder, num_cols, scaled=True)
    X_test_scaled, _ = preprocess_features(df_test, scaler, encoder, num_cols, scaled=True)
    
    X_train_unscaled, _ = preprocess_features(df_train, scaler, encoder, num_cols, scaled=False)
    X_test_unscaled, _ = preprocess_features(df_test, scaler, encoder, num_cols, scaled=False)
    
    # Targets
    y_train_reg = df_train['popularity'].values
    y_test_reg = df_test['popularity'].values
    
    y_train_clf_bin = df_train['skipped'].values
    y_test_clf_bin = df_test['skipped'].values
    
    y_train_clf_multi = df_train['skip_risk_category'].values
    y_test_clf_multi = df_test['skip_risk_category'].values
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_unscaled': X_train_unscaled,
        'X_test_unscaled': X_test_unscaled,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf_bin': y_train_clf_bin,
        'y_test_clf_bin': y_test_clf_bin,
        'y_train_clf_multi': y_train_clf_multi,
        'y_test_clf_multi': y_test_clf_multi,
        'feature_names': feature_names,
        'num_cols': num_cols
    }

if __name__ == '__main__':
    # Test execution
    data = prepare_data_and_preprocessors(sample_size=30000)
    print("Done Preprocessing. Train feature matrix shape:", data['X_train_scaled'].shape)
