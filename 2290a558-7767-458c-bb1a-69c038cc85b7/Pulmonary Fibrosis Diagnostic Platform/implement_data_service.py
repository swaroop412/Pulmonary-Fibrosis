import os

# Create services/data_service.py - Data processing service
data_service_content = '''"""
Data Service - Handles file preview and dataset statistics
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional


class DataService:
    """Service for data file processing and analysis"""
    
    def __init__(self):
        """Initialize data service"""
        self.data_dir = Path("data")
        self.supported_formats = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
    
    def _read_file(self, file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Read data file into pandas DataFrame
        
        Args:
            file_path: Path to the file
            nrows: Number of rows to read (None for all)
        
        Returns:
            pandas DataFrame
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path, nrows=nrows)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=nrows)
        elif suffix == '.json':
            df = pd.read_json(file_path)
            return df.head(nrows) if nrows else df
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
            return df.head(nrows) if nrows else df
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def preview_file(self, file_path: str, num_rows: int = 5) -> Dict[str, Any]:
        """
        Preview a data file
        
        Args:
            file_path: Relative path to file in data directory
            num_rows: Number of rows to preview (default: 5)
        
        Returns:
            Dict with preview data including columns, sample rows, and basic info
        """
        try:
            full_path = self.data_dir / file_path
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}"
                }
            
            if full_path.suffix.lower() not in self.supported_formats:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {full_path.suffix}. Supported: {', '.join(self.supported_formats)}"
                }
            
            # Read file
            df = self._read_file(full_path, nrows=num_rows)
            
            # Convert preview rows to dict
            preview_data = df.to_dict(orient='records')
            
            return {
                "status": "success",
                "file_path": file_path,
                "columns": list(df.columns),
                "num_columns": len(df.columns),
                "preview_rows": num_rows,
                "data": preview_data,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading file: {str(e)}"
            }
    
    def get_statistics(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a dataset
        
        Args:
            file_path: Relative path to file in data directory
        
        Returns:
            Dict with dataset statistics including row count, columns, missing values, data types
        """
        try:
            full_path = self.data_dir / file_path
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}"
                }
            
            if full_path.suffix.lower() not in self.supported_formats:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {full_path.suffix}"
                }
            
            # Read full file
            df = self._read_file(full_path)
            
            # Calculate statistics
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Missing values per column
            missing_values = df.isnull().sum().to_dict()
            missing_percentages = (df.isnull().sum() / total_rows * 100).round(2).to_dict()
            
            # Column info
            columns_info = []
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(missing_values[col]),
                    "missing_percentage": float(missing_percentages[col]),
                    "unique_values": int(df[col].nunique())
                }
                
                # Add numeric statistics if applicable
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update({
                        "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                        "median": float(df[col].median()) if not df[col].isna().all() else None,
                        "min": float(df[col].min()) if not df[col].isna().all() else None,
                        "max": float(df[col].max()) if not df[col].isna().all() else None,
                        "std": float(df[col].std()) if not df[col].isna().all() else None
                    })
                
                columns_info.append(col_info)
            
            # Overall statistics
            total_missing = df.isnull().sum().sum()
            total_cells = total_rows * total_cols
            overall_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
            
            # File size
            file_size_bytes = full_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            return {
                "status": "success",
                "file_path": file_path,
                "file_size_mb": round(file_size_mb, 2),
                "row_count": total_rows,
                "column_count": total_cols,
                "total_cells": total_cells,
                "total_missing_values": int(total_missing),
                "missing_percentage": round(overall_missing_pct, 2),
                "columns": columns_info,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing file: {str(e)}"
            }
    
    def list_files(self) -> Dict[str, Any]:
        """
        List all data files in the data directory
        
        Returns:
            Dict with list of available files
        """
        try:
            files = []
            for file_path in self.data_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    relative_path = str(file_path.relative_to(self.data_dir))
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    files.append({
                        "path": relative_path,
                        "name": file_path.name,
                        "format": file_path.suffix.lower(),
                        "size_mb": round(file_size_mb, 2)
                    })
            
            return {
                "status": "success",
                "files": files,
                "total_files": len(files)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing files: {str(e)}",
                "files": []
            }
'''

with open('services/data_service.py', 'w') as f:
    f.write(data_service_content)
print("✓ Created services/data_service.py")

print("\n✅ Data service implementation complete!")
print("\nService features:")
print("  • File preview with configurable row count")
print("  • Comprehensive dataset statistics")
print("  • Missing values analysis per column")
print("  • Data type detection and validation")
print("  • Numeric column statistics (mean, median, min, max, std)")
print("  • File size and memory usage reporting")
print("  • Support for CSV, Excel, JSON, and Parquet formats")
print("  • List all available data files")

data_service_status = "complete"
