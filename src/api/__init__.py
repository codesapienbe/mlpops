from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import time
import asyncio
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

from config import config_manager, setup_logger, PredictionError
from core import Predictor
from pipeline import PipelineOrchestrator

# Metrics
REQUEST_COUNTER = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_TIME = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])
PREDICTION_GAUGE = Gauge('api_predictions_total', 'Total predictions made', ['type'])

logger = setup_logger('api')

class JobInput(BaseModel):
    job_title: str
    company: str
    location: str
    description: str
    requirements: str
    posted_date: str
    
    class Config:
        str_strip_whitespace = True
        str_min_length = 1

class BatchInput(BaseModel):
    jobs: List[JobInput]
    
    class Config:
        str_max_length = 1000

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = config_manager.config
    app_state['predictor'] = Predictor(config)
    app_state['config'] = config
    logger.info(f"API startup complete on {config['api']['host']}:{config['api']['port']}")
    yield
    # Shutdown
    logger.info(f"API shutdown complete on {config['api']['host']}:{config['api']['port']}")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Analysis API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            
        duration = time.time() - start_time
        endpoint = request.url.path
        
        REQUEST_COUNTER.labels(
            method=request.method,
            endpoint=endpoint,
            status=status_code
        ).inc()
        REQUEST_TIME.labels(endpoint=endpoint).observe(duration)
        
        logger.info(f"{request.method} {endpoint} - {status_code} - {duration:.3f}s")
        return response
    
    @app.middleware("http")
    async def validate_requests(request: Request, call_next):
        if request.url.path in ['/health', '/metrics']:
            return await call_next(request)
            
        if len(request.url.path) > 100 or not request.url.path.startswith('/'):
            logger.warning(f"Rejected invalid request: {request.url.path}")
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid request")
            
        return await call_next(request)
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error", "errors": exc.errors()}
        )
    
    @app.exception_handler(PredictionError)
    async def prediction_exception_handler(request: Request, exc: PredictionError):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction failed: {str(exc)}"}
        )
    
    @app.post("/analyze", response_model=PredictionResponse)
    async def analyze(job: JobInput) -> PredictionResponse:
        logger.info(f"Received single analyze request: {job.dict()}")
        try:
            logger.debug("Starting single prediction")
            start_time = time.time()
            prediction = app_state['predictor'].predict_single(job)
            logger.info(f"Single prediction output: {prediction}")
            processing_time = time.time() - start_time
            
            PREDICTION_GAUGE.labels(type='single').inc()
            
            return PredictionResponse(
                prediction=float(prediction),
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f'Single prediction failed: {str(e)}', exc_info=True)
            raise PredictionError(str(e))
    
    @app.post("/batch-analyze")
    async def batch_analyze(batch: BatchInput):
        logger.info(f"Received batch analyze request with {len(batch.jobs)} jobs")
        for idx, job in enumerate(batch.jobs):
            logger.debug(f"Batch job {idx}: {job.dict()}")
        if len(batch.jobs) > 100:
            raise HTTPException(400, "Batch size too large (max 100)")
            
        try:
            logger.debug("Starting batch prediction")
            predictions = app_state['predictor'].predict_batch(batch.jobs)
            logger.info(f"Batch prediction output: {predictions}")
            PREDICTION_GAUGE.labels(type='batch').inc()
            return {"predictions": list(predictions)}
        except Exception as e:
            logger.error(f'Batch prediction failed: {str(e)}', exc_info=True)
            raise PredictionError(str(e))
    
    @app.get('/health')
    async def health():
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get('/metrics')
    async def metrics():
        return generate_latest(REGISTRY)
    
    @app.post("/evaluate")
    async def evaluate_pipeline():
        """Run the full pipeline and return stage results"""
        try:
            orchestrator = PipelineOrchestrator(app_state['config'])
            success = orchestrator.run()
            # Serialize results
            serialized = {}
            for stage, result in orchestrator.results.items():
                serialized[stage.value] = {
                    "success": result.success,
                    "duration": result.duration,
                    "error": result.error,
                    "metrics": result.metrics
                }
            return {"success": success, "results": serialized}
        except Exception as e:
            logger.error(f"Pipeline evaluation failed: {str(e)}", exc_info=True)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Evaluation failed: {str(e)}")
    
    return app

def main():
    import uvicorn
    app = create_app()
    config = config_manager.config
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'], reload=False)

if __name__ == "__main__":
    main()
