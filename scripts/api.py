import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.utils.logger import logger
    from src.utils.exception import CustomException
    from src.pipeline.inference_pipeline import InferencePipeline
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    raise

# Global pipeline instance
pipeline = None

def initialize_pipeline():
    """Initialize the inference pipeline"""
    global pipeline
    try:
        logger.info("Initializing InferencePipeline...")
        
        # Try different possible config paths
        config_paths = [
            "configs/inference_pipeline.yaml",
            "configs/pipeline_params.yaml", 
            os.path.join(project_root, "configs", "inference_pipeline.yaml"),
            os.path.join(project_root, "configs", "pipeline_params.yaml")
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            # Create default config
            default_config = {
                "inference_pipeline": {
                    "model_path": os.path.join(project_root, "artifacts", "best_model", "best.h5"),
                    "clean_config_path": os.path.join(project_root, "configs", "config.yaml"),
                    "batch_separator": "|||",
                    "save_results": False,
                    "output_path": os.path.join(project_root, "artifacts", "inference", "api_results.csv")
                }
            }
            # Save default config
            os.makedirs(os.path.join(project_root, "configs"), exist_ok=True)
            config_path = os.path.join(project_root, "configs", "inference_pipeline.yaml")
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
            logger.info(f"Created default config at: {config_path}")
        
        logger.info(f"Using config: {config_path}")
        pipeline = InferencePipeline(config_path=config_path)
        logger.info(f"InferencePipeline initialized successfully with model: {pipeline.model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        pipeline = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Sentiment Analysis API...")
    initialize_pipeline()
    yield
    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using trained ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SinglePredictionRequest(BaseModel):
    title: str = Field(..., description="Product title or review title")
    text: Optional[str] = Field(None, description="Review text (optional)")

class BatchPredictionRequest(BaseModel):
    items: List[Dict[str, str]] = Field(..., description="List of title-text pairs")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted sentiment")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    title: str = Field(..., description="Original title")
    text: Optional[str] = Field(None, description="Original text")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return {
        "status": "healthy" if pipeline else "unhealthy",
        "model_loaded": pipeline is not None,
        "model_type": getattr(pipeline, 'model_type', None) if pipeline else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """
    Predict sentiment for a single review
    """
    try:
        if not pipeline:
            if not initialize_pipeline():
                raise HTTPException(status_code=503, detail="Inference pipeline not initialized")
        
        logger.info(f"Single prediction request - Title: {request.title[:50]}...")
        
        # Run inference
        result_df = pipeline.run(
            title=request.title,
            text=request.text,
            batch_mode=False
        )
        
        # Extract results
        prediction = result_df["predicted_label"].iloc[0]
        
        response = PredictionResponse(
            prediction=prediction,
            title=request.title,
            text=request.text,
            confidence=0.95  # Placeholder
        )
        
        logger.info(f"Prediction: {prediction}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test the API with a simple endpoint first
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {"message": "API is working!", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0 for Windows
        port=8000,
        reload=False,
        log_level="info"
    )