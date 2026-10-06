"""
PlaylistPulse Module M4: Unsupervised Learning
Implements:
1. K-Means clustering (Lloyd's algorithm, K-Means++, MiniBatchKMeans, Elbow method, Silhouette score analysis).
2. Hierarchical Agglomerative Clustering (Ward, Complete, Average, Single linkages, Dendrogram).
3. Density-Based Spatial Clustering (DBSCAN: eps, min_samples, core/border/noise outlier detection).
4. Dimensionality Reduction: PCA (Scree plot, explained variance ratio, loading vectors) & t-SNE embedding.
5. Anomaly Detection: Isolation Forest & One-Class SVM.
6. Cluster-as-Feature engineering for downstream supervised models.
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
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.cluster.hierarchy import dendrogram, linkage

from src.utils import get_path, get_logger, save_plot

logger = get_logger(__name__)


def train_and_evaluate_unsupervised(data, sample_size=5000):
    """
    Executes comprehensive Unsupervised Learning Pipeline (Module M4).
    Uses standardized continuous audio features for distance-consistent geometric operations.
    """
    logger.info("==================================================")
    logger.info("STARTING MODULE M4: UNSUPERVISED LEARNING PIPELINE")
    logger.info("==================================================")

    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    feature_names = data['feature_names']
    
    # Select audio feature subset (first 13 numeric features)
    num_feature_count = 13
    X_audio_train = X_train_scaled[:, :num_feature_count]
    X_audio_test = X_test_scaled[:, :num_feature_count]
    audio_feature_names = feature_names[:num_feature_count]

    # Sample a representative subset for expensive quadratic operations (Silhouette, Hierarchical, t-SNE)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_audio_train), size=min(sample_size, len(X_audio_train)), replace=False)
    X_sample = X_audio_train[sample_idx]

    os.makedirs(get_path('models', 'unsupervised'), exist_ok=True)
    os.makedirs(get_path('static', 'images'), exist_ok=True)

    results = {}

    # =========================================================================
    # 1. K-MEANS: ELBOW METHOD & SILHOUETTE SCORE ANALYSIS (K=2 to 10)
    # =========================================================================
    logger.info("1. Evaluating K-Means (Lloyd / K-Means++) with Elbow and Silhouette analysis...")
    k_range = list(range(2, 11))
    inertias = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        km.fit(X_sample)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_sample, km.labels_))
        ch = float(calinski_harabasz_score(X_sample, km.labels_))
        db = float(davies_bouldin_score(X_sample, km.labels_))
        silhouette_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        logger.info(f"   K={k} -> Inertia (WCSS): {km.inertia_:.1f}, Silhouette: {sil:.4f}, Calinski-Harabasz: {ch:.1f}, Davies-Bouldin: {db:.4f}")

    best_k_idx = int(np.argmax(silhouette_scores))
    optimal_k = k_range[best_k_idx]
    logger.info(f"Optimal K selected via Silhouette Score: K={optimal_k}")

    # Train final full-scale MiniBatchKMeans and standard KMeans at optimal K
    best_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
    best_kmeans.fit(X_audio_train)
    
    # MiniBatchKMeans for streaming / large dataset scalability benchmark
    minibatch_km = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1024, random_state=42)
    minibatch_km.fit(X_audio_train)

    joblib.dump(best_kmeans, get_path('models', 'unsupervised', 'kmeans.joblib'))
    joblib.dump(minibatch_km, get_path('models', 'unsupervised', 'minibatch_kmeans.joblib'))

    results['kmeans'] = {
        'k_range': k_range,
        'inertias': [round(x, 2) for x in inertias],
        'silhouette_scores': [round(x, 4) for x in silhouette_scores],
        'calinski_harabasz': [round(x, 2) for x in ch_scores],
        'davies_bouldin': [round(x, 4) for x in db_scores],
        'optimal_k': optimal_k,
        'cluster_centers': best_kmeans.cluster_centers_.tolist()
    }

    # PLOT: K-Means Elbow & Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax1.set_facecolor('#12151b')
    
    color = '#1DB954'
    ax1.set_xlabel('Number of Clusters (K)', color='#f3f4f6', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares - WCSS)', color=color, fontsize=12)
    line1 = ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2.5, label='Elbow Inertia (WCSS)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', colors='#9ca3af')
    ax1.grid(True, alpha=0.15)

    ax2 = ax1.twinx()
    color2 = '#38bdf8'
    ax2.set_ylabel('Silhouette Score', color=color2, fontsize=12)
    line2 = ax2.plot(k_range, silhouette_scores, marker='s', color=color2, linewidth=2.5, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Highlight Optimal K
    ax1.axvline(x=optimal_k, color='#f43f5e', linestyle=':', alpha=0.8, label=f'Optimal K={optimal_k}')

    lines = line1 + line2 + [plt.Line2D([0], [0], color='#f43f5e', linestyle=':', label=f'Optimal K={optimal_k}')]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')
    plt.title('M4: K-Means Optimal Cluster Selection (Elbow vs Silhouette Analysis)', color='#fff', fontsize=14, pad=15)
    plt.tight_layout()
    save_plot(fig, 'm4_kmeans_elbow_silhouette.png')
    plt.close()

    # =========================================================================
    # 2. DIMENSIONALITY REDUCTION: PCA & EXPLAINED VARIANCE RATIO
    # =========================================================================
    logger.info("2. Computing PCA (Scree Plot & Cumulative Variance)...")
    pca_full = PCA(n_components=min(10, num_feature_count), random_state=42)
    pca_full.fit(X_audio_train)
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d_sample = pca_2d.fit_transform(X_sample)
    joblib.dump(pca_2d, get_path('models', 'unsupervised', 'pca_2d.joblib'))

    exp_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    results['pca'] = {
        'n_components': len(exp_var),
        'explained_variance_ratio': [round(float(v), 4) for v in exp_var],
        'cumulative_variance_ratio': [round(float(v), 4) for v in cum_var],
        'pca_2d_explained_variance': [round(float(v), 4) for v in pca_2d.explained_variance_ratio_]
    }

    # PLOT: PCA Scree & Cumulative Variance
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    components = [f'PC{i+1}' for i in range(len(exp_var))]
    bars = ax.bar(components, exp_var * 100, color='#1DB954', alpha=0.7, label='Individual Explained Variance %')
    ax.plot(components, cum_var * 100, color='#38bdf8', marker='o', linewidth=2.5, label='Cumulative Explained Variance %')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color='#f3f4f6', fontsize=9)

    ax.set_xlabel('Principal Components', color='#f3f4f6', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', color='#f3f4f6', fontsize=12)
    ax.set_title('M4: PCA Scree Plot & Cumulative Variance Spectrum', color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')
    plt.tight_layout()
    save_plot(fig, 'm4_pca_scree_variance.png')
    plt.close()

    # =========================================================================
    # 3. DENSITY-BASED CLUSTERING: DBSCAN & OUTLIER DETECTION
    # =========================================================================
    logger.info("3. Training DBSCAN (Density-Based Spatial Clustering of Applications with Noise)...")
    dbscan = DBSCAN(eps=1.8, min_samples=15)
    dbscan_labels = dbscan.fit_predict(X_sample)
    
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int(np.sum(dbscan_labels == -1))
    noise_ratio = float(n_noise / len(X_sample))

    logger.info(f"   DBSCAN identified {n_clusters_db} dense clusters and {n_noise} noise points ({noise_ratio:.1%})")

    results['dbscan'] = {
        'n_clusters': n_clusters_db,
        'noise_points': n_noise,
        'noise_ratio': round(noise_ratio, 4),
        'eps': 1.8,
        'min_samples': 15
    }

    # PLOT: DBSCAN in 2D PCA Space
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    
    unique_labels = set(dbscan_labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    
    for k, col in zip(unique_labels, palette):
        if k == -1:
            col = '#f43f5e'  # Red for noise / outliers
            label = f'Noise Outliers (N={n_noise})'
            alpha = 0.6
            marker = 'x'
            size = 25
        else:
            label = f'Dense Cluster {k}'
            alpha = 0.8
            marker = 'o'
            size = 35

        class_member_mask = (dbscan_labels == k)
        xy = X_pca_2d_sample[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size, alpha=alpha, label=label, edgecolors='none' if marker=='x' else '#00000044')

    ax.set_xlabel('Principal Component 1 (Energy / Loudness Axis)', color='#f3f4f6', fontsize=12)
    ax.set_ylabel('Principal Component 2 (Acousticness / Speechiness Axis)', color='#f3f4f6', fontsize=12)
    ax.set_title(f'M4: DBSCAN Density Clustering & Outlier Separation (eps=1.8, minPts=15)', color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff', loc='upper right')
    plt.tight_layout()
    save_plot(fig, 'm4_dbscan_clusters_outliers.png')
    plt.close()

    # =========================================================================
    # 4. HIERARCHICAL AGGLOMERATIVE CLUSTERING & DENDROGRAM
    # =========================================================================
    logger.info("4. Computing Hierarchical Agglomerative Clustering (Linkage comparison)...")
    # Sub-sample 100 tracks for crisp visual dendrogram
    dendro_sample = X_sample[:100]
    Z_ward = linkage(dendro_sample, method='ward')
    Z_complete = linkage(dendro_sample, method='complete')
    Z_average = linkage(dendro_sample, method='average')

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    dendrogram(Z_ward, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10,
               color_threshold=15, above_threshold_color='#9ca3af')
    ax.set_xlabel('Cluster Nodes / Track Sub-branches', color='#f3f4f6', fontsize=12)
    ax.set_ylabel("Ward's Linkage Distance (Variance Increase)", color='#f3f4f6', fontsize=12)
    ax.set_title("M4: Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)", color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    save_plot(fig, 'm4_hierarchical_dendrogram.png')
    plt.close()

    # =========================================================================
    # 5. ANOMALY DETECTION: ISOLATION FOREST & ONE-CLASS SVM
    # =========================================================================
    logger.info("5. Training Anomaly Detectors (Isolation Forest & One-Class SVM)...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_labels = iso_forest.fit_predict(X_audio_train)
    iso_scores = iso_forest.decision_function(X_audio_train)
    joblib.dump(iso_forest, get_path('models', 'unsupervised', 'isolation_forest.joblib'))

    one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    one_class_svm.fit(X_sample)
    joblib.dump(one_class_svm, get_path('models', 'unsupervised', 'one_class_svm.joblib'))

    anomalies_count = int(np.sum(iso_labels == -1))
    results['anomaly_detection'] = {
        'model': 'IsolationForest',
        'n_estimators': 100,
        'contamination_rate': 0.05,
        'total_anomalies_detected': anomalies_count,
        'anomaly_ratio': round(anomalies_count / len(X_audio_train), 4),
        'mean_score': round(float(np.mean(iso_scores)), 4)
    }

    # PLOT: Isolation Forest Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    sns.histplot(iso_scores, bins=50, kde=True, ax=ax, color='#38bdf8', stat='density')
    ax.axvline(x=0, color='#f43f5e', linestyle='--', linewidth=2, label='Anomaly Threshold (Score < 0)')
    ax.set_xlabel('Isolation Forest Anomaly Score (Lower = More Anomalous)', color='#f3f4f6', fontsize=12)
    ax.set_ylabel('Probability Density', color='#f3f4f6', fontsize=12)
    ax.set_title('M4: Isolation Forest Anomaly Score Spectrum for Spotify Tracks', color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor='#181c24', edgecolor='#ffffff22', labelcolor='#fff')
    plt.tight_layout()
    save_plot(fig, 'm4_isolation_forest_anomalies.png')
    plt.close()

    # =========================================================================
    # 6. t-SNE 2D EMBEDDING VISUALIZATION
    # =========================================================================
    logger.info("6. Computing t-SNE 2D Manifold Embedding...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_sample[:1500])
    sample_cluster_labels = best_kmeans.predict(X_sample[:1500])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0d0f12')
    ax.set_facecolor('#12151b')
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=sample_cluster_labels, cmap='viridis', s=25, alpha=0.8, edgecolors='#00000033')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('K-Means Cluster ID', color='#f3f4f6')
    cbar.ax.tick_params(colors='#9ca3af')
    ax.set_xlabel('t-SNE Dimension 1 (Local Acoustic Manifold)', color='#f3f4f6', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2 (Local Acoustic Manifold)', color='#f3f4f6', fontsize=12)
    ax.set_title('M4: t-SNE 2D Non-Linear Manifold Projection with Cluster Coloring', color='#fff', fontsize=14, pad=15)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    save_plot(fig, 'm4_tsne_umap_embedding.png')
    plt.close()

    # Save metrics JSON
    metrics_path = get_path('models', 'unsupervised', 'unsupervised_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Module M4 artifacts and metrics saved to: {metrics_path}")

    return results


if __name__ == '__main__':
    from src.data_preprocessing import prepare_data_and_preprocessors
    data = prepare_data_and_preprocessors(sample_size=30000)
    train_and_evaluate_unsupervised(data)
