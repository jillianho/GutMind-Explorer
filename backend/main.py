"""
GutMind Explorer - FastAPI Backend
Real microbiome analysis API with ML predictions.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
import logging
import os
from io import StringIO
from pathlib import Path

from data_loader import (
    get_research_dataset, 
    load_user_data, 
    calculate_diversity_metrics,
    get_bacteria_columns,
    MENTAL_HEALTH_BACTERIA,
    PSYCHOBIOTIC_EFFECTS
)
from ml_models import (
    get_trained_model,
    MicrobiomeAnalyzer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GutMind Explorer API",
    description="Microbiome-Mental Health Analysis Platform",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class BacteriaProfile(BaseModel):
    """User's bacteria abundance profile."""
    Lactobacillus: float = 0
    Bifidobacterium: float = 0
    Bacteroides: float = 0
    Prevotella: float = 0
    Faecalibacterium: float = 0
    Roseburia: float = 0
    Akkermansia: float = 0
    Clostridium: float = 0
    Bilophila: float = 0
    Desulfovibrio: float = 0
    # Add more as needed

class PredictionRequest(BaseModel):
    """Request for ML prediction."""
    profile: Dict[str, float]
    target: str = "anxiety"  # "anxiety" or "depression"

class AnalysisRequest(BaseModel):
    """Request for statistical analysis."""
    analysis_type: str  # "correlations", "pca", "clustering"
    params: Optional[Dict[str, Any]] = None


# ============== Startup ==============

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup."""
    logger.info("Starting GutMind Explorer API...")
    
    # Load research dataset
    df = get_research_dataset()
    logger.info(f"Research dataset loaded: {len(df)} samples")
    
    # Train model
    model = get_trained_model()
    logger.info(f"ML model trained: AUC = {model.training_stats['ensemble']['auc_roc']}")


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "online",
        "api": "GutMind Explorer",
        "version": "2.0.0",
        "endpoints": [
            "/api/dataset/info",
            "/api/dataset/sample",
            "/api/predict",
            "/api/analyze/correlations",
            "/api/analyze/pca",
            "/api/analyze/clustering",
            "/api/compare",
            "/api/upload",
            "/api/bacteria/info"
        ]
    }


@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the research dataset."""
    df = get_research_dataset()
    bacteria_cols = get_bacteria_columns(df)
    
    return {
        "name": "Synthetic American Gut Project Dataset",
        "description": "Simulated microbiome data based on real AGP distributions with mental health correlations",
        "n_samples": len(df),
        "n_bacteria": len(bacteria_cols),
        "bacteria_list": bacteria_cols,
        "mental_health_measures": ["anxiety_score", "depression_score", "anxiety_level", "depression_level"],
        "demographics": {
            "age_range": [int(df['age'].min()), int(df['age'].max())],
            "sex_distribution": df['sex'].value_counts().to_dict()
        },
        "anxiety_distribution": {
            "mean": round(float(df['anxiety_score'].mean()), 1),
            "std": round(float(df['anxiety_score'].std()), 1),
            "pct_high": round(float((df['anxiety_level'] == 'high').mean() * 100), 1)
        },
        "depression_distribution": {
            "mean": round(float(df['depression_score'].mean()), 1),
            "std": round(float(df['depression_score'].std()), 1),
            "pct_high": round(float((df['depression_level'] == 'high').mean() * 100), 1)
        }
    }


@app.get("/api/dataset/sample")
async def get_dataset_sample(n: int = Query(default=10, le=100)):
    """Get a sample of the research dataset."""
    df = get_research_dataset()
    sample = df.sample(min(n, len(df))).to_dict(orient='records')
    
    # Round floats for cleaner output
    for record in sample:
        for key, value in record.items():
            if isinstance(value, float):
                record[key] = round(value, 3)
    
    return {"sample": sample, "n": len(sample)}


@app.get("/api/dataset/full")
async def get_full_dataset():
    """Get the full research dataset for frontend visualization."""
    df = get_research_dataset()
    bacteria_cols = get_bacteria_columns(df)
    
    # Prepare data for frontend
    subjects = []
    for _, row in df.iterrows():
        subject = {
            'id': row['sample_id'],
            'bacteria': {col: round(row[col], 3) for col in bacteria_cols},
            'anxietyScore': float(row['anxiety_score']),
            'depressionScore': float(row['depression_score']),
            'anxietyLevel': row['anxiety_level'],
            'depressionLevel': row['depression_level'],
            'age': int(row['age']),
            'sex': row['sex']
        }
        subjects.append(subject)
    
    return {
        'bacteria': bacteria_cols,
        'subjects': subjects,
        'psychobioticEffects': PSYCHOBIOTIC_EFFECTS
    }


@app.post("/api/predict")
async def predict_mental_health(request: PredictionRequest):
    """
    Predict mental health outcome from microbiome profile.
    Uses ensemble ML model trained on research data.
    """
    try:
        # Convert profile to dataframe
        user_df = pd.DataFrame([request.profile])
        
        # Get prediction
        model = get_trained_model()
        prediction = model.predict(user_df)
        
        # Add model info
        prediction['model_info'] = {
            'type': 'Ensemble (Random Forest + Gradient Boosting)',
            'training_samples': model.training_stats['n_samples'],
            'auc_roc': model.training_stats['ensemble']['auc_roc'],
            'cv_auc': model.training_stats['cv_auc_mean']
        }
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/correlations")
async def analyze_correlations(target: str = Query(default="anxiety_score")):
    """Get bacteria-mental health correlations."""
    df = get_research_dataset()
    
    valid_targets = ['anxiety_score', 'depression_score']
    if target not in valid_targets:
        raise HTTPException(status_code=400, detail=f"Target must be one of {valid_targets}")
    
    correlations = MicrobiomeAnalyzer.calculate_correlations(df, target)
    
    return {
        "target": target,
        "correlations": correlations,
        "significant_count": sum(1 for c in correlations if c['significant']),
        "top_positive": [c for c in correlations if c['pearson_r'] > 0][:5],
        "top_negative": [c for c in correlations if c['pearson_r'] < 0][:5]
    }


@app.get("/api/analyze/pca")
async def analyze_pca(n_components: int = Query(default=5, le=20)):
    """Run PCA on the microbiome data."""
    df = get_research_dataset()
    result = MicrobiomeAnalyzer.run_pca(df, n_components)
    
    # Add metadata
    result['n_samples'] = len(df)
    result['mental_health'] = {
        'anxiety_levels': df['anxiety_level'].tolist(),
        'depression_levels': df['depression_level'].tolist(),
        'anxiety_scores': df['anxiety_score'].tolist()
    }
    
    return result


@app.get("/api/analyze/clustering")
async def analyze_clustering(n_clusters: int = Query(default=3, ge=2, le=10)):
    """Run K-means clustering on microbiome data."""
    df = get_research_dataset()
    result = MicrobiomeAnalyzer.run_clustering(df, n_clusters)
    
    return result


@app.post("/api/compare")
async def compare_to_population(profile: Dict[str, float]):
    """Compare user's profile to the research population."""
    try:
        user_df = pd.DataFrame([profile])
        population_df = get_research_dataset()
        
        comparison = MicrobiomeAnalyzer.compare_to_population(user_df, population_df)
        
        # Add diversity metrics
        comparison['diversity'] = calculate_diversity_metrics(user_df)
        
        return comparison
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_microbiome_data(file: UploadFile = File(...)):
    """
    Upload and parse microbiome test results.
    Supports: Biomesight CSV, generic CSV formats.
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse the file
        user_df, detected_format = load_user_data(content_str)
        
        # Get research dataset for comparison
        population_df = get_research_dataset()
        
        # Compare to population
        comparison = MicrobiomeAnalyzer.compare_to_population(user_df, population_df)
        
        # Get prediction
        model = get_trained_model()
        prediction = model.predict(user_df)
        
        # Calculate diversity
        diversity = calculate_diversity_metrics(user_df)
        
        return {
            "status": "success",
            "filename": file.filename,
            "detected_format": detected_format,
            "parsed_bacteria": user_df.to_dict(orient='records')[0],
            "comparison": comparison,
            "prediction": prediction,
            "diversity": diversity
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")


@app.get("/api/bacteria/info")
async def get_bacteria_info(name: Optional[str] = None):
    """Get information about bacteria and their mental health associations."""
    if name:
        info = PSYCHOBIOTIC_EFFECTS.get(name)
        if info:
            return {"bacteria": name, **info}
        else:
            return {"bacteria": name, "effect": "unknown", "confidence": "none"}
    else:
        return {
            "bacteria_info": PSYCHOBIOTIC_EFFECTS,
            "key_bacteria": MENTAL_HEALTH_BACTERIA
        }


@app.get("/api/model/info")
async def get_model_info():
    """Get information about the trained ML model."""
    model = get_trained_model()
    
    return {
        "model_type": "Ensemble (Random Forest + Gradient Boosting)",
        "is_trained": model.is_trained,
        "training_stats": model.training_stats,
        "feature_importance": model.training_stats.get('feature_importance', [])[:10]
    }


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# ============== Serve Frontend ==============

# Serve the frontend HTML
@app.get("/app")
async def serve_frontend():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"error": "Frontend not found"}


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
