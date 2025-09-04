"""
MLOps Job Analysis Package
"""
__version__ = "1.0.0"

import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Centralized configuration management"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            # Prefer explicit CONFIG_PATH if provided; fallback to repo root config.yml
            env_config_path = os.environ.get('CONFIG_PATH')
            if env_config_path:
                config_path = Path(env_config_path)
            else:
                root = Path(__file__).resolve().parents[2]
                config_path = root / "config.yml"
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    @property
    def config(self) -> Dict[str, Any]:
        return self.load_config()

# Helper to resolve project root consistently (works locally and in Docker)
def get_project_root() -> Path:
    """Return the project root derived from CONFIG_PATH or package location."""
    env_config_path = os.environ.get('CONFIG_PATH')
    if env_config_path:
        return Path(env_config_path).resolve().parent
    return Path(__file__).resolve().parents[2]

# Global config instance
config_manager = ConfigManager()

class MLOpsException(Exception):
    """Base exception for MLOps pipeline"""
    pass

class DataValidationError(MLOpsException):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(MLOpsException):
    """Raised when model training fails"""
    pass

class PredictionError(MLOpsException):
    """Raised when prediction fails"""
    pass

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup structured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger
