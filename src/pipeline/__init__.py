import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from prometheus_client import start_http_server, Summary, Gauge, Counter
import mlflow
from pathlib import Path

from config import config_manager, setup_logger
from config import get_project_root
from core import RobustIngestion, LanguageAwareCleaner, MultiLangTrainer

# Metrics
TRAIN_TIME = Summary('pipeline_train_seconds', 'Time spent training model')
DATA_GAUGE = Gauge('pipeline_data_rows', 'Number of rows processed', ['stage'])
ERROR_COUNTER = Counter('pipeline_errors_total', 'Total pipeline errors', ['stage'])

logger = setup_logger('pipeline')

class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    INGESTION = "ingestion"
    CLEANING = "cleaning"
    TRAINING = "training"
    EVALUATION = "evaluation"

@dataclass
class StageResult:
    """Result of a pipeline stage"""
    stage: PipelineStage
    success: bool
    duration: float
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class PipelineTelemetry:
    """Pipeline telemetry and monitoring"""
    
    def __init__(self):
        self.start_time = None
        self.current_stage = None
    
    def stage_start(self, stage_name):
        self.current_stage = stage_name
        logger.info(f"Stage START: {stage_name}")
        return self

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            logger.error(f"Stage FAILED: {self.current_stage} ({duration:.2f}s)")
            ERROR_COUNTER.labels(stage=self.current_stage).inc()
            return False
        logger.info(f"Stage COMPLETE: {self.current_stage} ({duration:.2f}s)")
        return True

class PipelineOrchestrator:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results: Dict[PipelineStage, StageResult] = {}
        # Pipeline data storage
        self.train_df = None
        self.test_df = None
        self.cleaned_train_df = None
        self.cleaned_test_df = None
        self.trainer = None
    
    def run(self) -> bool:
        """Run complete pipeline"""
        stages = [
            (PipelineStage.INGESTION, self._run_ingestion),
            (PipelineStage.CLEANING, self._run_cleaning),
            (PipelineStage.TRAINING, self._run_training),
            (PipelineStage.EVALUATION, self._run_evaluation),
        ]
        
        for stage, stage_func in stages:
            if not self._execute_stage(stage, stage_func):
                logger.error(f'Pipeline failed at {stage.value}')
                return False
        
        logger.info('Pipeline completed successfully')
        return True
    
    def _execute_stage(self, stage: PipelineStage, stage_func) -> bool:
        """Execute a single pipeline stage"""
        start_time = time.time()
        try:
            result = stage_func()
            duration = time.time() - start_time
            
            self.results[stage] = StageResult(
                stage=stage,
                success=True,
                duration=duration,
                metrics=result if isinstance(result, dict) else None
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.results[stage] = StageResult(
                stage=stage,
                success=False,
                duration=duration,
                error=str(e)
            )
            logger.error(f'Stage {stage.value} failed: {str(e)}')
            return False
    
    def _run_ingestion(self):
        """Run data ingestion stage"""
        ingestion = RobustIngestion(self.config)
        # Ingest and split data
        self.train_df, self.test_df = ingestion.load_data()
        DATA_GAUGE.labels('raw').set(len(self.train_df))
        return {"train_rows": len(self.train_df), "test_rows": len(self.test_df)}
    
    def _run_cleaning(self):
        """Run data cleaning stage"""
        # Ensure ingestion stage has run
        if self.train_df is None or self.test_df is None:
            raise RuntimeError("Ingestion must run before cleaning")
        cleaner = LanguageAwareCleaner(self.config)
        # Clean both train and test sets
        self.cleaned_train_df = cleaner.clean_data(self.train_df)
        self.cleaned_test_df = cleaner.clean_data(self.test_df)
        DATA_GAUGE.labels('cleaned').set(len(self.cleaned_train_df))
        return {"cleaned_train_rows": len(self.cleaned_train_df), "cleaned_test_rows": len(self.cleaned_test_df)}
    
    def _run_training(self):
        """Run model training stage"""
        # Ensure cleaning stage has run
        if self.cleaned_train_df is None:
            raise RuntimeError("Cleaning must run before training")
        # Train and save model
        self.trainer = MultiLangTrainer(self.config)
        X = self.cleaned_train_df[['experience_years', 'remote', 'clean_description']]
        y = self.cleaned_train_df[self.config['model']['target']]
        # Log model training parameters to MLflow
        mlflow.log_params(self.config.get('model', {}).get('params', {}))
        self.trainer.train_and_save(X, y)
        # Log trained model artifact to MLflow
        model_cfg = self.config.get('model', {})
        project_root = get_project_root()
        model_path = project_root / model_cfg.get('store_path', 'models') / model_cfg.get('file_name', 'model.pkl')
        mlflow.log_artifact(str(model_path), artifact_path="model")
        return {"trained_rows": len(self.cleaned_train_df)}

    def _run_evaluation(self):
        """Run model evaluation stage"""
        # Ensure training stage has run
        if self.trainer is None or self.cleaned_test_df is None:
            raise RuntimeError("Training must run before evaluation")
        # Prepare test data
        X_test = self.cleaned_test_df[['experience_years', 'remote', 'clean_description']]
        y_test = self.cleaned_test_df[self.config['model']['target']]
        # Evaluate model
        accuracy, report, roc_auc = self.trainer.evaluate_model(X_test, y_test)
        # Log evaluation metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        logger.info(f"Evaluation metrics: accuracy={accuracy}, roc_auc={roc_auc}")
        return {"accuracy": accuracy, "roc_auc": roc_auc, "report": report}

@TRAIN_TIME.time()
def run_pipeline():
    """Main pipeline execution function"""
    config = config_manager.config
    # Start an MLflow run for the pipeline
    with mlflow.start_run():
        try:
            logger.info("🚀 Starting pipeline execution")
            start_time = time.time()

            # Data Ingestion
            with PipelineTelemetry().stage_start("ingestion"):
                train_df, test_df = RobustIngestion(config).load_data()
                DATA_GAUGE.labels('raw').set(len(train_df))

            # Data Cleaning
            with PipelineTelemetry().stage_start("cleaning"):
                cleaner = LanguageAwareCleaner(config)
                cleaned_df = cleaner.clean_data(train_df)
                test_cleaned = cleaner.clean_data(test_df)
                DATA_GAUGE.labels('cleaned').set(len(cleaned_df))

            # Model Training
            with PipelineTelemetry().stage_start("training"):
                trainer = MultiLangTrainer(config)
                X = cleaned_df[['experience_years', 'remote', 'clean_description']]
                y = cleaned_df[config['model']['target']]
                # Log model training parameters to MLflow
                mlflow.log_params(config.get('model', {}).get('params', {}))
                trainer.train_and_save(X, y)
                # Log trained model artifact to MLflow
                model_cfg = config.get('model', {})
                project_root = get_project_root()
                model_path = project_root / model_cfg.get('store_path', 'models') / model_cfg.get('file_name', 'model.pkl')
                mlflow.log_artifact(str(model_path), artifact_path="model")

            # Model Evaluation
            with PipelineTelemetry().stage_start("evaluation"):
                X_test = test_cleaned[['experience_years', 'remote', 'clean_description']]
                y_test = test_cleaned[config['model']['target']]
                accuracy, report, roc_auc = trainer.evaluate_model(X_test, y_test)
                # Log evaluation metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("roc_auc", roc_auc)
                logger.info(f"Evaluation metrics: accuracy={accuracy}, roc_auc={roc_auc}")

            logger.info(f"✅ Pipeline completed in {time.time()-start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            raise

def main():
    """Entry point for the pipeline script"""
    start_http_server(8001)
    run_pipeline()

if __name__ == "__main__":
    main()
