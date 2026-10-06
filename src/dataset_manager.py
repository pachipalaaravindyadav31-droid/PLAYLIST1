"""
PlaylistPulse Dataset Management Service
Handles:
- Uploading and saving new CSV datasets
- Schema validation and feature compatibility checks
- Active dataset tracking and metadata extraction
- Summary statistics and preview generation for UI
"""
import os
import json
import datetime
import pandas as pd
import numpy as np

from src.utils import get_path, get_logger

logger = get_logger(__name__)

CONFIG_PATH = get_path('data', 'dataset_config.json')
UPLOADS_DIR = get_path('data', 'uploads')
DEFAULT_SAMPLE_PATH = get_path('data', 'clean_sample.csv')

REQUIRED_AUDIO_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

RECOMMENDED_COLS = [
    'popularity', 'duration_ms', 'key', 'mode', 'time_signature', 'explicit', 'track_genre'
]

OPTIONAL_TEXT_COLS = ['track_name', 'artists', 'album_name']


def init_dataset_config():
    """Initializes the dataset configuration file if it does not exist."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        active_path = DEFAULT_SAMPLE_PATH
        config = {
            'active_dataset': active_path,
            'active_name': 'Default Spotify Sample (5,000 tracks)',
            'updated_at': datetime.datetime.now().isoformat(),
            'datasets': [
                {
                    'name': 'Default Spotify Sample',
                    'filename': 'clean_sample.csv',
                    'path': DEFAULT_SAMPLE_PATH,
                    'uploaded_at': datetime.datetime.now().isoformat(),
                    'is_default': True
                }
            ]
        }
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        return config
    else:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_active_dataset_path():
    """Returns the absolute path to the active dataset."""
    config = init_dataset_config()
    active_path = config.get('active_dataset', DEFAULT_SAMPLE_PATH)
    if not os.path.exists(active_path):
        # Fallback to default
        active_path = DEFAULT_SAMPLE_PATH
    return active_path


def get_dataset_summary(filepath=None):
    """
    Reads a dataset and returns structured metadata, validation info,
    summary statistics, and first 10 rows for preview.
    """
    if filepath is None:
        filepath = get_active_dataset_path()

    if not os.path.exists(filepath):
        return {
            'error': f'Dataset file not found at {filepath}',
            'is_valid': False
        }

    try:
        df = pd.read_csv(filepath)
        row_count, col_count = df.shape
        file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
        filename = os.path.basename(filepath)

        # Check required & optional columns
        present_cols = set(df.columns)
        missing_required = [c for c in REQUIRED_AUDIO_COLS if c not in present_cols]
        missing_recommended = [c for c in RECOMMENDED_COLS if c not in present_cols]

        is_valid = len(missing_required) == 0

        # Column detail list
        column_details = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / row_count) * 100, 2) if row_count > 0 else 0
            
            is_req = col in REQUIRED_AUDIO_COLS
            is_rec = col in RECOMMENDED_COLS
            
            column_details.append({
                'name': col,
                'dtype': dtype_str,
                'null_count': null_count,
                'null_pct': null_pct,
                'category': 'Required Audio' if is_req else ('Recommended' if is_rec else 'Metadata')
            })

        # Summary statistics for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary_stats = {}
        for col in num_cols:
            summary_stats[col] = {
                'mean': round(float(df[col].mean()), 4),
                'std': round(float(df[col].std()), 4) if not np.isnan(df[col].std()) else 0.0,
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4),
                'median': round(float(df[col].median()), 4)
            }

        # Preview table (first 10 rows)
        preview_df = df.head(10).fillna('N/A')
        preview_records = preview_df.to_dict(orient='records')
        preview_columns = preview_df.columns.tolist()

        return {
            'is_valid': is_valid,
            'filename': filename,
            'filepath': filepath,
            'file_size_kb': file_size_kb,
            'row_count': row_count,
            'col_count': col_count,
            'missing_required': missing_required,
            'missing_recommended': missing_recommended,
            'column_details': column_details,
            'summary_stats': summary_stats,
            'preview_columns': preview_columns,
            'preview_records': preview_records
        }

    except Exception as e:
        logger.error(f"Error extracting dataset summary: {e}", exc_info=True)
        return {
            'error': str(e),
            'is_valid': False
        }


def save_and_activate_dataset(file_obj, filename, make_active=True):
    """
    Saves an uploaded CSV file, validates its columns, and updates dataset configuration.
    """
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.basename(filename).replace(" ", "_")
    saved_filename = f"{timestamp}_{clean_name}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)

    # Save file
    file_obj.save(saved_path)
    logger.info(f"Uploaded dataset saved to: {saved_path}")

    # Validate dataframe
    try:
        df = pd.read_csv(saved_path)
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return False, f"Invalid CSV file format: {str(e)}", None

    if len(df) < 50:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return False, f"Dataset contains too few records ({len(df)} rows). Minimum 50 rows required for ML training.", None

    summary = get_dataset_summary(saved_path)
    if not summary.get('is_valid', False):
        missing = ", ".join(summary.get('missing_required', []))
        # We don't necessarily delete it, but warn
        logger.warning(f"Uploaded dataset is missing required columns: {missing}")

    # Update config
    config = init_dataset_config()
    new_entry = {
        'name': clean_name,
        'filename': saved_filename,
        'path': saved_path,
        'uploaded_at': datetime.datetime.now().isoformat(),
        'rows': len(df),
        'cols': len(df.columns),
        'is_default': False
    }
    config.setdefault('datasets', []).append(new_entry)

    if make_active:
        config['active_dataset'] = saved_path
        config['active_name'] = clean_name
        config['updated_at'] = datetime.datetime.now().isoformat()

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return True, "Dataset uploaded and registered successfully!", summary


def list_all_datasets():
    """Returns the list of all available datasets registered in the system."""
    config = init_dataset_config()
    active_path = config.get('active_dataset', DEFAULT_SAMPLE_PATH)
    datasets = config.get('datasets', [])
    
    # Enrich with live file existence
    for d in datasets:
        d['exists'] = os.path.exists(d.get('path', ''))
        d['is_active'] = (d.get('path') == active_path)
        if d['exists']:
            d['size_kb'] = round(os.path.getsize(d['path']) / 1024, 2)
            
    return datasets, active_path


def set_active_dataset(dataset_path):
    """Switches the active dataset to the specified path."""
    if not os.path.exists(dataset_path):
        return False, f"File not found: {dataset_path}"

    config = init_dataset_config()
    config['active_dataset'] = dataset_path
    config['active_name'] = os.path.basename(dataset_path)
    config['updated_at'] = datetime.datetime.now().isoformat()

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    logger.info(f"Active dataset switched to: {dataset_path}")
    return True, "Active dataset successfully updated."
