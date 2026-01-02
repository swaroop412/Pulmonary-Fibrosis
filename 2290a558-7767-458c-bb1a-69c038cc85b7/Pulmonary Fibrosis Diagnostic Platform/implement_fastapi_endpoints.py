import os

# Create app/routers.py - API endpoints
routers_content = '''"""
API Routers - FastAPI endpoints for dataset operations
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.kaggle_service import KaggleService
from services.data_service import DataService

# Initialize routers
router = APIRouter()

# Initialize services
kaggle_service = KaggleService()
data_service = DataService()


# Request/Response Models
class DownloadRequest(BaseModel):
    """Request model for dataset download"""
    dataset_id: str = Field(..., description="Kaggle dataset ID (e.g., 'username/dataset-name')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "datasnaek/youtube-new"
            }
        }


class PreviewRequest(BaseModel):
    """Request model for file preview"""
    file_path: str = Field(..., description="Relative path to file in data directory")
    num_rows: Optional[int] = Field(5, description="Number of rows to preview", ge=1, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "datasnaek_youtube-new/USvideos.csv",
                "num_rows": 10
            }
        }


class StatisticsRequest(BaseModel):
    """Request model for dataset statistics"""
    file_path: str = Field(..., description="Relative path to file in data directory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "datasnaek_youtube-new/USvideos.csv"
            }
        }


# API Endpoints

@router.post("/download", tags=["Dataset Operations"])
async def download_dataset(request: DownloadRequest):
    """
    Download a dataset from Kaggle
    
    - **dataset_id**: Kaggle dataset identifier (format: username/dataset-name)
    
    Returns information about the downloaded files and their location.
    
    **Note**: Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables to be set.
    """
    result = kaggle_service.download_dataset(request.dataset_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.post("/preview", tags=["Data Operations"])
async def preview_file(request: PreviewRequest):
    """
    Preview a data file
    
    - **file_path**: Relative path to file in data directory
    - **num_rows**: Number of rows to preview (default: 5, max: 100)
    
    Returns column names, data types, and sample rows.
    """
    result = data_service.preview_file(request.file_path, request.num_rows)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result


@router.post("/statistics", tags=["Data Operations"])
async def get_statistics(request: StatisticsRequest):
    """
    Get comprehensive statistics for a dataset
    
    - **file_path**: Relative path to file in data directory
    
    Returns detailed statistics including:
    - Row count and column count
    - Missing values per column
    - Data types
    - Numeric statistics (mean, median, min, max, std)
    - File size and memory usage
    """
    result = data_service.get_statistics(request.file_path)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result


@router.get("/datasets", tags=["Dataset Operations"])
async def list_datasets():
    """
    List all downloaded datasets
    
    Returns information about all datasets in the data directory.
    """
    result = kaggle_service.list_downloaded_datasets()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@router.get("/files", tags=["Data Operations"])
async def list_files():
    """
    List all available data files
    
    Returns a list of all supported data files in the data directory
    with file sizes and formats.
    """
    result = data_service.list_files()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result
'''

with open('app/routers.py', 'w') as f:
    f.write(routers_content)
print("✓ Created app/routers.py")

# Update app/main.py to include the routers
updated_main_content = '''"""
FastAPI Main Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import routers
from app.routers import router

app = FastAPI(
    title="Kaggle Dataset API",
    version="1.0.0",
    description="API for downloading Kaggle datasets and analyzing data files",
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

# Include API routers
app.include_router(router, prefix="/api/v1")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Welcome to Kaggle Dataset API!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "download": "/api/v1/download",
            "preview": "/api/v1/preview",
            "statistics": "/api/v1/statistics",
            "list_datasets": "/api/v1/datasets",
            "list_files": "/api/v1/files"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    kaggle_configured = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    
    return {
        "status": "healthy",
        "kaggle_configured": kaggle_configured,
        "warning": "Kaggle credentials not configured" if not kaggle_configured else None
    }
'''

with open('app/main.py', 'w') as f:
    f.write(updated_main_content)
print("✓ Updated app/main.py with API routers")

print("\n✅ FastAPI endpoints implementation complete!")
print("\nAPI Endpoints created:")
print("  POST /api/v1/download - Download Kaggle datasets")
print("  POST /api/v1/preview - Preview data files")
print("  POST /api/v1/statistics - Get dataset statistics")
print("  GET  /api/v1/datasets - List downloaded datasets")
print("  GET  /api/v1/files - List available data files")
print("  GET  / - API welcome and documentation")
print("  GET  /health - Health check endpoint")
print("\nFeatures:")
print("  ✓ Proper error handling with HTTP status codes")
print("  ✓ Request/response models with Pydantic")
print("  ✓ Comprehensive API documentation (Swagger UI)")
print("  ✓ CORS middleware for web access")
print("  ✓ Environment variable configuration")

api_status = "complete"
