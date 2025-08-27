# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
import joblib
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
import uvicorn
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Env-driven config
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]

def get_allowed_origins() -> List[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS))
    return [o.strip() for o in raw.split(",") if o.strip()]

API_PORT = int(os.environ.get("PORT", os.environ.get("API_PORT", "5001")))

# Initialize FastAPI app
app = FastAPI(
    title="Lead Scoring API",
    description="ML-powered lead scoring prediction service",
    version="2.0.0",
    docs_url="/v2/docs",
    redoc_url="/v2/redoc"
)

# Add CORS middleware with restricted origins via env
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_loaded = False
preprocessor_loaded = False
predictions_count = 0

# Expected feature columns (must match training data)
EXPECTED_COLUMNS = [
    'Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'TotalVisits',
    'Total Time Spent on Website', 'Page Views Per Visit', 'Last Activity',
    'Country', 'Specialization', 'What is your current occupation', 'Search',
    'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement',
    'Through Recommendations', 'Tags', 'Lead Quality', 'City',
    'A free copy of Mastering The Interview', 'Last Notable Activity'
]

NUMERIC_COLUMNS = [
    'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'
]

# Pydantic models for request/response
class LeadScoringRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "TotalVisits": 5,
                "Page Views Per Visit": 3.2,
                "Total Time Spent on Website": 1850,
                "Lead Origin": "API",
                "Lead Source": "Google",
                "Last Activity": "Email Opened",
                "What is your current occupation": "Working Professional"
            }
        }
    )

    TotalVisits: int = Field(..., ge=0, description="Number of website visits")
    Page_Views_Per_Visit: float = Field(..., ge=0, alias="Page Views Per Visit", description="Average page views per visit")
    Total_Time_Spent_on_Website: int = Field(..., ge=0, alias="Total Time Spent on Website", description="Total time spent on website in seconds")
    Lead_Origin: str = Field(..., alias="Lead Origin", description="Origin of the lead")
    Lead_Source: str = Field(..., alias="Lead Source", description="Source of the lead")
    Last_Activity: str = Field(..., alias="Last Activity", description="Last recorded activity")
    What_is_your_current_occupation: str = Field(..., alias="What is your current occupation", description="Current occupation")

    # Optional fields with defaults (categorical only)
    Do_Not_Email: str = Field("No", alias="Do Not Email")
    Do_Not_Call: str = Field("No", alias="Do Not Call")
    Country: str = Field("India", alias="Country")
    Specialization: str = Field("Not Specified", alias="Specialization")
    Search: str = Field("No", alias="Search")
    Newspaper_Article: str = Field("No", alias="Newspaper Article")
    X_Education_Forums: str = Field("No", alias="X Education Forums")
    Newspaper: str = Field("No", alias="Newspaper")
    Digital_Advertisement: str = Field("No", alias="Digital Advertisement")
    Through_Recommendations: str = Field("No", alias="Through Recommendations")
    Tags: str = Field("Not Specified", alias="Tags")
    Lead_Quality: str = Field("Not Specified", alias="Lead Quality")
    City: str = Field("Mumbai", alias="City")
    A_free_copy_of_Mastering_The_Interview: str = Field("No", alias="A free copy of Mastering The Interview")
    Last_Notable_Activity: str = Field("Modified", alias="Last Notable Activity")

    @field_validator('TotalVisits')
    def validate_total_visits(cls, v):
        if v < 0:
            raise ValueError('Total visits must be non-negative')
        return v

    @field_validator('Page_Views_Per_Visit')
    def validate_page_views(cls, v):
        if v < 0:
            raise ValueError('Page views per visit must be non-negative')
        return v

    @field_validator('Total_Time_Spent_on_Website')
    def validate_time_spent(cls, v):
        if v < 0:
            raise ValueError('Total time spent must be non-negative')
        return v

class LeadScoringResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="Prediction result (0 or 1)")
    lead_score: float = Field(..., description="Lead score percentage")
    label: str = Field(..., description="Human readable prediction label")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    predictions_count: int
    timestamp: str
    version: str
    uptime: Optional[str] = None

# Startup event to load models
@app.on_event("startup")
async def load_models():
    global model, preprocessor, model_loaded, preprocessor_loaded

    try:
        # Load model
        model_path = "models/model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_loaded = True
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")

        # Load preprocessor
        preprocessor_path = "models/preprocessor.joblib"
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            preprocessor_loaded = True
            logger.info("✅ Preprocessor loaded successfully")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")

        if model_loaded and preprocessor_loaded:
            logger.info("🚀 Lead Scoring API is ready!")
        else:
            logger.warning("API started but models not fully loaded (health will be degraded)")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")


def increment_predictions_count():
    global predictions_count
    predictions_count += 1


def prepare_features(request_data: LeadScoringRequest) -> pd.DataFrame:
    """Convert request data to DataFrame with all expected columns, keeping numeric types correct."""
    data_dict: Dict[str, Any] = {}

    # Map the request fields to the expected column names
    field_mapping = {
        'TotalVisits': 'TotalVisits',
        'Page_Views_Per_Visit': 'Page Views Per Visit',
        'Total_Time_Spent_on_Website': 'Total Time Spent on Website',
        'Lead_Origin': 'Lead Origin',
        'Lead_Source': 'Lead Source',
        'Last_Activity': 'Last Activity',
        'What_is_your_current_occupation': 'What is your current occupation',
        'Do_Not_Email': 'Do Not Email',
        'Do_Not_Call': 'Do Not Call',
        'Country': 'Country',
        'Specialization': 'Specialization',
        'Search': 'Search',
        'Newspaper_Article': 'Newspaper Article',
        'X_Education_Forums': 'X Education Forums',
        'Newspaper': 'Newspaper',
        'Digital_Advertisement': 'Digital Advertisement',
        'Through_Recommendations': 'Through Recommendations',
        'Tags': 'Tags',
        'Lead_Quality': 'Lead Quality',
        'City': 'City',
        'A_free_copy_of_Mastering_The_Interview': 'A free copy of Mastering The Interview',
        'Last_Notable_Activity': 'Last Notable Activity'
    }

    request_dict = request_data.model_dump(by_alias=False)

    for field_name, column_name in field_mapping.items():
        if field_name in request_dict:
            data_dict[column_name] = request_dict[field_name]

    # Ensure all expected columns are present
    for col in EXPECTED_COLUMNS:
        if col not in data_dict:
            # For numeric columns use 0; for categorical use "Not Specified"
            if col in NUMERIC_COLUMNS:
                data_dict[col] = 0
            else:
                data_dict[col] = 'Not Specified'

    df = pd.DataFrame([data_dict])

    # Cast numeric
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Reorder columns
    df = df[EXPECTED_COLUMNS]
    return df

@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Lead Scoring API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/v2/predict",
            "health": "/v2/health",
            "docs": "/v2/docs"
        }
    }

@app.get("/v2/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check():
    status = "healthy" if (model_loaded and preprocessor_loaded) else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        predictions_count=predictions_count,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.post("/v2/predict", response_model=LeadScoringResponse, summary="Predict lead conversion")
async def predict_lead_conversion(
    request: LeadScoringRequest,
    background_tasks: BackgroundTasks
):
    if not model_loaded or not preprocessor_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check server logs.")

    try:
        features_df = prepare_features(request)
        features_processed = preprocessor.transform(features_df)
        prediction = model.predict(features_processed)[0]
        prediction_proba = model.predict_proba(features_processed)[0]
        lead_score = float(prediction_proba[1] * 100)
        label = "Will Convert" if prediction == 1 else "Will Not Convert"
        background_tasks.add_task(increment_predictions_count)
        return LeadScoringResponse(
            prediction=int(prediction),
            lead_score=round(lead_score, 2),
            label=label,
            timestamp=datetime.now().isoformat(),
            model_version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/v2/models/info", summary="Model information")
async def get_model_info():
    info = {
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "expected_features": EXPECTED_COLUMNS,
        "feature_count": len(EXPECTED_COLUMNS)
    }
    if model_loaded and hasattr(model, 'classes_'):
        info["model_classes"] = model.classes_.tolist()
    return info

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
        log_level="info"
    )