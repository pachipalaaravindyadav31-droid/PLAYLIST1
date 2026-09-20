import os
import logging

def get_logger(name):
    """Set up and return a standardized logger that writes to both file and console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_path(*path_segments):
    """Get absolute path relative to the project root directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(project_root, *path_segments)

def save_plot(plt, filename):
    """Save a matplotlib plot to the static/images directory."""
    image_dir = get_path('static', 'images')
    os.makedirs(image_dir, exist_ok=True)
    filepath = os.path.join(image_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close()
    return filepath
