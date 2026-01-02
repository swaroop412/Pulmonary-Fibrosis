"""
FastAPI Prediction Endpoints for Model Inference
Creates API routes for multimodal FVC prediction with authentication
"""
import os

print("=" * 70)
print("FASTAPI PREDICTION ENDPOINTS")
print("=" * 70)

# API Endpoints for Model Prediction
print("\n🔧 CREATING PREDICTION API ENDPOINTS")
print("-" * 70)

prediction_routers_content = '''"""
Prediction API Routers - FastAPI endpoints for FVC prediction
"""
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Header
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import sys
from pathlib import Path
import base64

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.model_service import ModelInferenceService

# Initialize router
prediction_router = APIRouter(prefix="/predict", tags=["Model Prediction"])

# Initialize model service
model_service = ModelInferenceService()


# Request/Response Models
class LabData(BaseModel):
    """Lab data for patient"""
    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (Male/Female)")
    smoking_status: str = Field(..., description="Smoking status")
    baseline_fvc: float = Field(..., ge=500, le=6000, description="Baseline FVC in ml")
    baseline_percent: Optional[float] = Field(75.0, ge=0, le=150, description="Baseline FVC percent predicted")
    
    @validator('sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male', 'female', 'm', 'f']:
            raise ValueError('Sex must be Male or Female')
        return v.capitalize() if v.lower() in ['male', 'female'] else ('Male' if v.lower() == 'm' else 'Female')
    
    @validator('smoking_status')
    def validate_smoking(cls, v):
        valid_statuses = ['Never smoked', 'Ex-smoker', 'Currently smokes']
        if v not in valid_statuses:
            raise ValueError(f'Smoking status must be one of: {valid_statuses}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 68,
                "sex": "Male",
                "smoking_status": "Ex-smoker",
                "baseline_fvc": 2800,
                "baseline_percent": 72.0
            }
        }


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    lab_data: Optional[LabData] = Field(None, description="Patient lab data and demographics")
    clinical_text: Optional[str] = Field(None, description="Clinical notes text")
    confidence_level: Optional[float] = Field(0.95, ge=0.8, le=0.99, description="Confidence level for intervals")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PATIENT_001",
                "lab_data": {
                    "age": 68,
                    "sex": "Male",
                    "smoking_status": "Ex-smoker",
                    "baseline_fvc": 2800,
                    "baseline_percent": 72.0
                },
                "clinical_text": "Patient presents with progressive dyspnea and reduced FVC.",
                "confidence_level": 0.95
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    status: str
    patient_id: Optional[str]
    prediction: dict
    confidence_interval: dict
    risk_stratification: dict
    model_info: dict


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    patients: List[PredictionRequest] = Field(..., description="List of patient data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patients": [
                    {
                        "patient_id": "PATIENT_001",
                        "lab_data": {
                            "age": 68,
                            "sex": "Male",
                            "smoking_status": "Ex-smoker",
                            "baseline_fvc": 2800,
                            "baseline_percent": 72.0
                        }
                    },
                    {
                        "patient_id": "PATIENT_002",
                        "lab_data": {
                            "age": 72,
                            "sex": "Female",
                            "smoking_status": "Never smoked",
                            "baseline_fvc": 2200,
                            "baseline_percent": 68.0
                        }
                    }
                ]
            }
        }


# Authentication helper (simple API key)
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """
    Verify API key for authentication
    In production: use proper authentication (OAuth2, JWT, etc.)
    """
    # For demo: accept any key or no key
    # In production: validate against database
    return True


# API Endpoints

@prediction_router.post("/", response_model=PredictionResponse)
async def predict_fvc_decline(
    request: PredictionRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    Predict FVC decline for a single patient
    
    Supports multimodal input:
    - **lab_data**: Patient demographics and lab values
    - **clinical_text**: Clinical notes
    - **image**: CT scan (via separate upload endpoint)
    
    Returns:
    - FVC decline rate prediction (ml/week)
    - Confidence intervals
    - Risk stratification (low/moderate/high/severe)
    - Model confidence score
    
    **Authentication**: Optional API key via X-API-Key header
    """
    # Verify authentication
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Validate input
    validation = model_service.validate_input(
        lab_data=request.lab_data.dict() if request.lab_data else None,
        image_data=None,  # Image handled separately
        clinical_text=request.clinical_text
    )
    
    if not validation['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {', '.join(validation['errors'])}"
        )
    
    # Preprocess inputs
    lab_emb = None
    if request.lab_data:
        lab_emb = model_service.preprocess_lab_data(request.lab_data.dict())
    
    text_emb = None
    if request.clinical_text:
        text_emb = model_service.preprocess_text(request.clinical_text)
    
    # Predict
    prediction_result = model_service.predict_fvc_decline(
        lab_embedding=lab_emb,
        image_embedding=None,
        text_embedding=text_emb,
        confidence_level=request.confidence_level
    )
    
    return {
        'status': 'success',
        'patient_id': request.patient_id,
        **prediction_result
    }


@prediction_router.post("/with-image")
async def predict_with_image(
    patient_id: Optional[str] = Form(None),
    age: int = Form(...),
    sex: str = Form(...),
    smoking_status: str = Form(...),
    baseline_fvc: float = Form(...),
    baseline_percent: Optional[float] = Form(75.0),
    clinical_text: Optional[str] = Form(None),
    ct_image: UploadFile = File(...),
    confidence_level: Optional[float] = Form(0.95),
    x_api_key: Optional[str] = Header(None)
):
    """
    Predict FVC decline with CT image upload
    
    Accepts:
    - Lab data as form fields
    - CT scan image as file upload
    - Optional clinical text
    
    **Image Requirements**:
    - Format: DICOM, PNG, or JPEG
    - Max size: 10MB
    - CT scan of lungs
    
    **Authentication**: Optional API key via X-API-Key header
    """
    # Verify authentication
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Read image data
    image_data = await ct_image.read()
    
    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Image file too large (max 10MB)")
    
    # Prepare lab data
    lab_data = {
        'age': age,
        'sex': sex,
        'smoking_status': smoking_status,
        'baseline_fvc': baseline_fvc,
        'baseline_percent': baseline_percent
    }
    
    # Validate input
    validation = model_service.validate_input(
        lab_data=lab_data,
        image_data=image_data,
        clinical_text=clinical_text
    )
    
    if not validation['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {', '.join(validation['errors'])}"
        )
    
    # Preprocess inputs
    lab_emb = model_service.preprocess_lab_data(lab_data)
    image_emb = model_service.preprocess_image(image_data)
    
    text_emb = None
    if clinical_text:
        text_emb = model_service.preprocess_text(clinical_text)
    
    # Predict
    prediction_result = model_service.predict_fvc_decline(
        lab_embedding=lab_emb,
        image_embedding=image_emb,
        text_embedding=text_emb,
        confidence_level=confidence_level
    )
    
    return {
        'status': 'success',
        'patient_id': patient_id,
        'image_filename': ct_image.filename,
        **prediction_result
    }


@prediction_router.post("/batch", response_model=List[PredictionResponse])
async def batch_predict(
    request: BatchPredictionRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    Batch prediction for multiple patients
    
    Processes multiple patients in a single request.
    Each patient can have different combinations of modalities.
    
    Returns predictions for all patients, with individual error handling.
    
    **Max patients per request**: 100
    **Authentication**: Optional API key via X-API-Key header
    """
    # Verify authentication
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if len(request.patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 patients per batch request"
        )
    
    results = []
    
    for patient_request in request.patients:
        try:
            # Validate
            validation = model_service.validate_input(
                lab_data=patient_request.lab_data.dict() if patient_request.lab_data else None,
                image_data=None,
                clinical_text=patient_request.clinical_text
            )
            
            if not validation['valid']:
                results.append({
                    'status': 'error',
                    'patient_id': patient_request.patient_id,
                    'prediction': {},
                    'confidence_interval': {},
                    'risk_stratification': {},
                    'model_info': {'error': ', '.join(validation['errors'])}
                })
                continue
            
            # Preprocess
            lab_emb = None
            if patient_request.lab_data:
                lab_emb = model_service.preprocess_lab_data(patient_request.lab_data.dict())
            
            text_emb = None
            if patient_request.clinical_text:
                text_emb = model_service.preprocess_text(patient_request.clinical_text)
            
            # Predict
            prediction_result = model_service.predict_fvc_decline(
                lab_embedding=lab_emb,
                image_embedding=None,
                text_embedding=text_emb,
                confidence_level=patient_request.confidence_level
            )
            
            results.append({
                'status': 'success',
                'patient_id': patient_request.patient_id,
                **prediction_result
            })
            
        except Exception as e:
            results.append({
                'status': 'error',
                'patient_id': patient_request.patient_id,
                'prediction': {},
                'confidence_interval': {},
                'risk_stratification': {},
                'model_info': {'error': str(e)}
            })
    
    return results


@prediction_router.get("/model-info", tags=["System"])
async def get_model_info():
    """
    Get model information and capabilities
    
    Returns:
    - Model version
    - Supported modalities
    - Input requirements
    - Performance metrics
    """
    return {
        'model_version': model_service.model_version,
        'architecture': 'Attention-Based Multimodal Fusion',
        'modalities': {
            'lab_data': {
                'required_fields': ['age', 'sex', 'smoking_status', 'baseline_fvc'],
                'optional_fields': ['baseline_percent'],
                'embedding_dim': model_service.lab_dim
            },
            'ct_image': {
                'formats': ['DICOM', 'PNG', 'JPEG'],
                'max_size_mb': 10,
                'embedding_dim': model_service.ct_dim
            },
            'clinical_text': {
                'max_length': 2000,
                'embedding_dim': model_service.text_dim
            }
        },
        'risk_thresholds': model_service.risk_thresholds,
        'performance': {
            'mean_mae': '2.42 ml/week',
            'mean_r2': '0.493',
            'cross_validation_folds': 5
        },
        'endpoints': {
            'single_prediction': '/predict/',
            'with_image': '/predict/with-image',
            'batch_prediction': '/predict/batch',
            'model_info': '/predict/model-info'
        }
    }
'''

with open('app/routers.py', 'w') as f:
    f.write(prediction_routers_content)

print("✓ Created app/routers.py with prediction endpoints")

# Update main.py to include prediction router
print("\n🔧 UPDATING MAIN APPLICATION")
print("-" * 70)

updated_main_py = '''"""
FastAPI Main Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import routers
from app.routers import prediction_router

app = FastAPI(
    title="FVC Prediction API",
    version="1.0.0",
    description="""
    **Multimodal FVC Decline Prediction API**
    
    This API provides endpoints for predicting Forced Vital Capacity (FVC) decline 
    in pulmonary fibrosis patients using multimodal machine learning.
    
    ## Features
    - 🔬 Multimodal analysis (lab data, CT images, clinical text)
    - 📊 Confidence intervals and risk stratification
    - 🚀 Batch processing support
    - 🔐 API key authentication (optional)
    
    ## Model Information
    - **Architecture**: Attention-Based Multimodal Fusion
    - **Input Modalities**: Lab data, CT scans, Clinical notes
    - **Output**: FVC decline rate (ml/week) with confidence intervals
    - **Risk Categories**: Low, Moderate, High, Severe
    
    ## Quick Start
    1. Use `/predict/` for single patient predictions
    2. Use `/predict/with-image` for predictions with CT scans
    3. Use `/predict/batch` for multiple patients
    4. Check `/predict/model-info` for model details
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(prediction_router, prefix="/api/v1")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass  # Static directory might not exist


@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Welcome to FVC Prediction API!",
        "version": "1.0.0",
        "model": "Multimodal Attention-Based Fusion",
        "docs": "/docs",
        "api_base": "/api/v1",
        "endpoints": {
            "single_prediction": "/api/v1/predict/",
            "with_image": "/api/v1/predict/with-image",
            "batch_prediction": "/api/v1/predict/batch",
            "model_info": "/api/v1/predict/model-info"
        },
        "quick_start": {
            "example_request": {
                "patient_id": "PATIENT_001",
                "lab_data": {
                    "age": 68,
                    "sex": "Male",
                    "smoking_status": "Ex-smoker",
                    "baseline_fvc": 2800,
                    "baseline_percent": 72.0
                },
                "confidence_level": 0.95
            },
            "curl_example": "curl -X POST 'http://localhost:8000/api/v1/predict/' -H 'Content-Type: application/json' -d '{...}'"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "api_version": "1.0.0"
    }
'''

with open('app/main.py', 'w') as f:
    f.write(updated_main_py)

print("✓ Updated app/main.py")

# Create test script
print("\n🧪 CREATING API TEST SCRIPT")
print("-" * 70)

test_script = '''"""
Test script for FVC Prediction API
Run this after starting the FastAPI server
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_single_prediction():
    """Test single patient prediction"""
    print("\\n" + "="*70)
    print("TEST 1: Single Patient Prediction (Lab Data Only)")
    print("="*70)
    
    payload = {
        "patient_id": "TEST_001",
        "lab_data": {
            "age": 68,
            "sex": "Male",
            "smoking_status": "Ex-smoker",
            "baseline_fvc": 2800,
            "baseline_percent": 72.0
        },
        "confidence_level": 0.95
    }
    
    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Prediction: {result['prediction']['fvc_decline_rate']} ml/week")
        print(f"✓ Risk: {result['risk_stratification']['category'].upper()}")
        print(f"✓ Confidence: {result['model_info']['confidence_score']}")
        print(f"✓ CI: [{result['confidence_interval']['lower_bound']}, {result['confidence_interval']['upper_bound']}]")
    else:
        print(f"✗ Error: {response.text}")

def test_multimodal_prediction():
    """Test prediction with clinical text"""
    print("\\n" + "="*70)
    print("TEST 2: Multimodal Prediction (Lab + Clinical Text)")
    print("="*70)
    
    payload = {
        "patient_id": "TEST_002",
        "lab_data": {
            "age": 72,
            "sex": "Female",
            "smoking_status": "Never smoked",
            "baseline_fvc": 2200,
            "baseline_percent": 68.0
        },
        "clinical_text": "Patient presents with progressive dyspnea, reduced FVC, and CT shows honeycombing pattern consistent with IPF.",
        "confidence_level": 0.95
    }
    
    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Prediction: {result['prediction']['fvc_decline_rate']} ml/week")
        print(f"✓ Risk: {result['risk_stratification']['category'].upper()}")
        print(f"✓ Modalities: {', '.join(result['model_info']['modalities_used'])}")
        print(f"✓ Confidence: {result['model_info']['confidence_score']}")
    else:
        print(f"✗ Error: {response.text}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\\n" + "="*70)
    print("TEST 3: Batch Prediction (3 Patients)")
    print("="*70)
    
    payload = {
        "patients": [
            {
                "patient_id": "BATCH_001",
                "lab_data": {
                    "age": 65,
                    "sex": "Male",
                    "smoking_status": "Ex-smoker",
                    "baseline_fvc": 2600,
                    "baseline_percent": 70.0
                }
            },
            {
                "patient_id": "BATCH_002",
                "lab_data": {
                    "age": 70,
                    "sex": "Female",
                    "smoking_status": "Never smoked",
                    "baseline_fvc": 2300,
                    "baseline_percent": 75.0
                }
            },
            {
                "patient_id": "BATCH_003",
                "lab_data": {
                    "age": 75,
                    "sex": "Male",
                    "smoking_status": "Currently smokes",
                    "baseline_fvc": 1900,
                    "baseline_percent": 62.0
                }
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Processed {len(results)} patients")
        for result in results:
            if result['status'] == 'success':
                print(f"  - {result['patient_id']}: {result['prediction']['fvc_decline_rate']} ml/week ({result['risk_stratification']['category']})")
            else:
                print(f"  - {result['patient_id']}: ERROR")
    else:
        print(f"✗ Error: {response.text}")

def test_model_info():
    """Test model info endpoint"""
    print("\\n" + "="*70)
    print("TEST 4: Model Information")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/predict/model-info")
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"✓ Model Version: {info['model_version']}")
        print(f"✓ Architecture: {info['architecture']}")
        print(f"✓ Performance: MAE={info['performance']['mean_mae']}, R²={info['performance']['mean_r2']}")
        print(f"✓ Modalities: {', '.join(info['modalities'].keys())}")
    else:
        print(f"✗ Error: {response.text}")

if __name__ == "__main__":
    print("\\n" + "#"*70)
    print("# FVC PREDICTION API TEST SUITE")
    print("#"*70)
    
    try:
        # Test all endpoints
        test_single_prediction()
        test_multimodal_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("\\n" + "="*70)
        print("✅ ALL TESTS COMPLETE")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\\n✗ Error: Could not connect to API server")
        print("Make sure the server is running: uvicorn app.main:app --reload")
'''

with open('test_api.py', 'w') as f:
    f.write(test_script)

print("✓ Created test_api.py")

# Update requirements
print("\n📦 UPDATING REQUIREMENTS")
print("-" * 70)

requirements_update = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
numpy==1.24.3
python-dotenv==1.0.0
requests==2.31.0
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements_update)

print("✓ Updated requirements.txt")

print("\n\n✅ FASTAPI PREDICTION ENDPOINTS COMPLETE")
print("=" * 70)
print("API Features:")
print("  ✓ POST /api/v1/predict/ - Single patient prediction")
print("  ✓ POST /api/v1/predict/with-image - Prediction with CT image upload")
print("  ✓ POST /api/v1/predict/batch - Batch prediction (up to 100 patients)")
print("  ✓ GET  /api/v1/predict/model-info - Model information")
print("  ✓ Input validation with Pydantic models")
print("  ✓ API key authentication (optional)")
print("  ✓ Comprehensive error handling")
print("  ✓ Auto-generated API documentation (Swagger UI)")
print("  ✓ CORS support for web access")
print("\nTo start the API:")
print("  uvicorn app.main:app --reload --port 8000")
print("\nTo test the API:")
print("  python test_api.py")
print("=" * 70)

api_ready = True
