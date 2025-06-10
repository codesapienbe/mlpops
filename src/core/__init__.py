import logging
import time
import functools
import re
import pandas as pd
import joblib
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from prometheus_client import Histogram

from config import setup_logger, DataValidationError, ModelTrainingError, PredictionError

# Download NLTK data
try:
    nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'], quiet=True)
except:
    pass

# Metrics
CORE_TIME = Histogram('core_function_duration_seconds', 'Core function duration', ['function'])

logger = setup_logger('core', logging.DEBUG)

def log_call(func):
    """Decorator for logging function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"▶️ Entering {func.__qualname__}")
        start_time = time.time()
        with CORE_TIME.labels(function=func.__qualname__).time():
            result = func(*args, **kwargs)
        logger.debug(f"⏹️ Exiting {func.__qualname__} ({time.time()-start_time:.3f}s)")
        return result
    return wrapper

class RobustIngestion:
    """Robust data ingestion with validation"""
    
    def __init__(self, config):
        self.config = config
        self.required_columns = {
            'job_title', 'company', 'location',
            'description', 'requirements', 'posted_date'
        }

    @log_call
    def _validate_schema(self, df):
        """Validate DataFrame schema"""
        missing = self.required_columns - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing columns: {missing}")
        return True

    @log_call
    def load_data(self):
        """Load and split data"""
        try:
            raw_path = Path(self.config['data']['raw_path']) / self.config['data']['jobs_file']
            
            if not raw_path.exists():
                raise DataValidationError(f"Data file not found: {raw_path}")
            
            df = pd.read_csv(raw_path)
            self._validate_schema(df)
            
            return train_test_split(df, **self.config['pipeline']['split'])
            
        except Exception as e:
            logger.error(f'Data loading failed: {str(e)}')
            raise DataValidationError(f"Failed to load data: {str(e)}")

class LanguageAwareCleaner:
    """Language-aware text cleaning and feature extraction"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')
        self._init_nlp_resources()

    @log_call
    def _init_nlp_resources(self):
        """Initialize NLP resources"""
        lang = self.config.get('language', 'english')
        try:
            self.stop_words = set(stopwords.words(lang))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(lang))
        
        self.stemmer = SnowballStemmer(lang) if lang in SnowballStemmer.languages else None

    @log_call
    def _clean_text(self, text):
        """Clean and process text"""
        if pd.isna(text):
            return ""
            
        text = re.sub(r'<[^>]+>|\s+', ' ', str(text)).lower()
        words = self.tokenizer.tokenize(text)
        
        if self.stemmer:
            words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        else:
            words = [w for w in words if w not in self.stop_words]
        
        return ' '.join(words)

    @log_call
    def _extract_features(self, df):
        """Extract features from text data"""
        lang_config = self.config.get('language_configs', {}).get(
            self.config.get('language', 'english'), {}
        )
        
        df = df.copy()
        df['clean_description'] = df['description'].apply(self._clean_text)
        
        # Extract experience (default pattern if not configured)
        exp_pattern = lang_config.get('experience_pattern', r'(\d+)\s*(?:years?|yrs?)')
        df['experience'] = df['requirements'].str.extract(exp_pattern).fillna(0).astype(int)
        
        # Extract remote keywords
        remote_keywords = lang_config.get('remote_keywords', ['remote', 'work from home', 'telecommute'])
        df['remote'] = df['description'].str.contains('|'.join(remote_keywords), case=False, na=False)
        
        return df

    @log_call
    def clean_data(self, df):
        """Main data cleaning pipeline"""
        try:
            df = self._extract_features(df)
            
            if df[['experience', 'remote']].isnull().sum().sum() > 0:
                logger.warning("Some null values found in extracted features")
                df['experience'] = df['experience'].fillna(0)
                df['remote'] = df['remote'].fillna(False)
            
            return df
            
        except Exception as e:
            logger.error(f'Data cleaning failed: {str(e)}')
            raise DataValidationError(f"Failed to clean data: {str(e)}")

class MultiLangTrainer:
    """Multi-language model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = self._build_pipeline()

    @log_call
    def _build_pipeline(self):
        """Build ML pipeline"""
        try:
            preprocessor = ColumnTransformer([
                ('minmax', MinMaxScaler(), ['experience']),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), ['remote'])
            ])

            return Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', RandomForestClassifier(**self.config.get('model', {}).get('params', {})))
            ])
        except Exception as e:
            logger.error(f'Pipeline building failed: {str(e)}')
            raise ModelTrainingError(f"Failed to build pipeline: {str(e)}")

    @log_call
    def train_and_save(self, X, y):
        """Train model and save to disk"""
        try:
            self.pipeline.fit(X, y)
            
            model_path = Path(self.config.get('model', {}).get('store_path', 'models'))
            model_path.mkdir(parents=True, exist_ok=True)
            file_path = model_path / 'model.pkl'
            
            joblib.dump(self.pipeline, file_path)
            logger.info(f"✅ Model saved to {file_path.resolve()}")
            
        except Exception as e:
            logger.error(f'Model training failed: {str(e)}')
            raise ModelTrainingError(f"Failed to train model: {str(e)}")

    @log_call
    def evaluate_model(self, X_test, y_test):
        """Evaluate trained model performance"""
        try:
            # Predictions
            y_pred = self.pipeline.predict(X_test)
            roc_auc = 0.0
            # Compute ROC-AUC if probabilities available
            if hasattr(self.pipeline, 'predict_proba'):
                y_score = self.pipeline.predict_proba(X_test)
                try:
                    n_classes = y_score.shape[1]
                    if n_classes == 2:
                        # Binary classification
                        roc_auc = roc_auc_score(y_test, y_score[:, 1])
                    else:
                        # Multiclass classification
                        roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC: {e}")
                    roc_auc = 0.0
            # Compute other metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            return accuracy, report, roc_auc
        except Exception as e:
            logger.error(f'Evaluation failed: {e}', exc_info=True)
            raise ModelTrainingError(f"Failed to evaluate model: {e}")

class Predictor:
    """Model predictor with fallback handling"""
    
    def __init__(self, config):
        model_config = config.get('model', {})
        store_path = model_config.get('store_path', 'models')
        model_file = model_config.get('file_name', 'model.pkl')
        # Resolve store_path relative to project root (two levels up from this file)
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / store_path / model_file
        
        # Create directory if needed
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Load model; disable fallback
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Model load error: {e}")
                self.pipeline = None
                logger.warning("Model could not be loaded; predictions are disabled")
        else:
            logger.warning(f"Model not found at {self.model_path}")
            self.pipeline = None
        
        self.cleaner = LanguageAwareCleaner(config)

    @log_call
    def _prepare_features(self, job):
        """Prepare features for single job"""
        if self.pipeline is None:
            raise PredictionError("Model not loaded; cannot perform prediction")
        try:
            df = pd.DataFrame([{
                'job_title': job.job_title,
                'company': job.company,
                'location': job.location,
                'description': job.description,
                'requirements': job.requirements,
                'posted_date': job.posted_date
            }])
            
            cleaned = self.cleaner.clean_data(df)
            return cleaned[['experience', 'remote']]
            
        except Exception as e:
            logger.error(f'Feature preparation failed: {str(e)}')
            raise PredictionError(f"Failed to prepare features: {str(e)}")

    @log_call
    def predict_single(self, job):
        """Predict single job"""
        if self.pipeline is None:
            raise PredictionError("Model not loaded; cannot perform prediction")
        try:
            X = self._prepare_features(job)
            return self.pipeline.predict(X)[0]
        except Exception as e:
            logger.error(f'Single prediction failed: {str(e)}')
            raise PredictionError(f"Failed to predict: {str(e)}")

    @log_call
    def predict_batch(self, jobs):
        """Predict batch of jobs"""
        if self.pipeline is None:
            raise PredictionError("Model not loaded; cannot perform batch prediction")
        try:
            df = pd.DataFrame([{
                'job_title': j.job_title,
                'company': j.company,
                'location': j.location,
                'description': j.description,
                'requirements': j.requirements,
                'posted_date': j.posted_date
            } for j in jobs])
            
            cleaned = self.cleaner.clean_data(df)
            X = cleaned[['experience', 'remote']]
            return self.pipeline.predict(X)
            
        except Exception as e:
            logger.error(f'Batch prediction failed: {str(e)}')
            raise PredictionError(f"Failed to predict batch: {str(e)}")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            y_pred = self.pipeline.predict(X_test)
            y_proba = self.pipeline.predict_proba(X_test)[:, 1] if hasattr(self.pipeline, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            report = classification_report(y_test, y_pred, output_dict=True)
            
            return accuracy, report, roc_auc
            
        except Exception as e:
            logger.error(f'Model evaluation failed: {str(e)}')
            return 0.0, {}, 0.0
