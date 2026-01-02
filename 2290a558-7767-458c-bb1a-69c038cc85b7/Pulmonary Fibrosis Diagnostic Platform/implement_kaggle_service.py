import os

# Create services/kaggle_service.py - Kaggle API integration
kaggle_service_content = '''"""
Kaggle Service - Handles Kaggle API integration for dataset downloads
"""
import os
import json
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import shutil


class KaggleService:
    """Service for interacting with Kaggle API"""
    
    def __init__(self):
        """Initialize Kaggle service with credentials from environment"""
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self._setup_credentials()
    
    def _setup_credentials(self):
        """Setup Kaggle API credentials from environment variables"""
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        
        if not kaggle_username or not kaggle_key:
            print("Warning: KAGGLE_USERNAME or KAGGLE_KEY not set in environment")
            return
        
        # Create .kaggle directory in home
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write credentials file
        kaggle_json = kaggle_dir / "kaggle.json"
        credentials = {
            "username": kaggle_username,
            "key": kaggle_key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set proper permissions (read/write for owner only)
        os.chmod(kaggle_json, 0o600)
    
    def download_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Download a Kaggle dataset
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., 'username/dataset-name')
        
        Returns:
            Dict with status, message, and downloaded files info
        """
        try:
            # Verify credentials are set
            if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
                return {
                    "status": "error",
                    "message": "Kaggle credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.",
                    "files": []
                }
            
            # Create dataset-specific directory
            dataset_name = dataset_id.replace('/', '_')
            dataset_path = self.data_dir / dataset_name
            dataset_path.mkdir(exist_ok=True)
            
            # Download using kaggle CLI
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_id, '-p', str(dataset_path), '--unzip'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Failed to download dataset: {result.stderr}",
                    "files": []
                }
            
            # List downloaded files
            downloaded_files = [str(f.relative_to(self.data_dir)) for f in dataset_path.rglob('*') if f.is_file()]
            
            return {
                "status": "success",
                "message": f"Successfully downloaded dataset: {dataset_id}",
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "files": downloaded_files,
                "file_count": len(downloaded_files)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "Download timeout - dataset too large or connection slow",
                "files": []
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "message": "Kaggle CLI not installed. Install with: pip install kaggle",
                "files": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "files": []
            }
    
    def list_downloaded_datasets(self) -> Dict[str, Any]:
        """
        List all downloaded datasets in data directory
        
        Returns:
            Dict with dataset information
        """
        try:
            datasets = []
            for item in self.data_dir.iterdir():
                if item.is_dir() and item.name != '.gitkeep':
                    files = [f.name for f in item.iterdir() if f.is_file()]
                    datasets.append({
                        "name": item.name,
                        "path": str(item),
                        "files": files,
                        "file_count": len(files)
                    })
            
            return {
                "status": "success",
                "datasets": datasets,
                "total_datasets": len(datasets)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing datasets: {str(e)}",
                "datasets": []
            }
'''

with open('services/kaggle_service.py', 'w') as f:
    f.write(kaggle_service_content)
print("✓ Created services/kaggle_service.py")

# Update requirements.txt to include kaggle
requirements_additional = '''kaggle==1.6.6
pandas==2.1.3
openpyxl==3.1.2
'''

with open('requirements.txt', 'a') as f:
    f.write(requirements_additional)
print("✓ Updated requirements.txt with kaggle, pandas, openpyxl")

# Update .env.example to include Kaggle credentials
env_update = '''
# Kaggle API Credentials
# Get these from https://www.kaggle.com/settings/account
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
'''

with open('.env.example', 'a') as f:
    f.write(env_update)
print("✓ Updated .env.example with Kaggle credentials")

print("\n✅ Kaggle service implementation complete!")
print("\nService features:")
print("  • Secure credential handling via environment variables")
print("  • Dataset download with automatic unzipping")
print("  • List downloaded datasets")
print("  • Proper error handling and timeouts")
print("  • Organized file storage in data/ directory")

kaggle_service_status = "complete"
