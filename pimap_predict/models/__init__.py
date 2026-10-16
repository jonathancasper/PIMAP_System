"""Model loading utilities for XGBoost prediction models.

Provides centralized model loading with schema validation.
Models are stored externally (not in source control) and follow
the naming convention: {model_name}_v{version}.{ext}
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class SchemaLoadError(Exception):
    """Raised when schema loading or validation fails."""
    pass


def load_model_schema(schema_path: Path) -> Dict:
    """Load and validate a model schema JSON file.
    
    Args:
        schema_path: Path to schema JSON file
        
    Returns:
        Schema dict with feature_names, feature_dtypes, etc.
        
    Raises:
        SchemaLoadError: If schema file not found or invalid
    """
    if not schema_path.exists():
        raise SchemaLoadError(f"Schema file not found: {schema_path}")
    
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise SchemaLoadError(f"Invalid JSON in schema file: {e}")
    
    required_keys = ['feature_names', 'n_features', 'target']
    for key in required_keys:
        if key not in schema:
            raise SchemaLoadError(f"Schema missing required key: {key}")
    
    if len(schema['feature_names']) != schema['n_features']:
        raise SchemaLoadError(
            f"Schema inconsistency: n_features={schema['n_features']} "
            f"but feature_names has {len(schema['feature_names'])} entries"
        )
    
    return schema


def resolve_model_paths(
    model_dir: Path,
    model_name: str,
    version: str,
) -> Tuple[Path, Path]:
    """Resolve model and schema file paths.
    
    Naming convention:
        Model: {model_name}_v{version}.json
        Schema: {model_name}_schema_v{version}.json
    
    Args:
        model_dir: Directory containing model files
        model_name: Base name of model (e.g., 'hapi_xgb')
        version: Version string (e.g., '1')
        
    Returns:
        Tuple of (model_path, schema_path)
    """
    model_file = f"{model_name}_v{version}.json"
    schema_file = f"{model_name}_schema_v{version}.json"
    
    return model_dir / model_file, model_dir / schema_file


def load_xgboost_model(
    model_dir: str,
    model_name: str = "hapi_xgb",
    version: str = "1",
) -> Tuple[object, Dict]:
    """Load XGBoost model and its schema.
    
    Args:
        model_dir: Path to directory containing model files
        model_name: Base name of model
        version: Version number
        
    Returns:
        Tuple of (xgboost_model, schema_dict)
        
    Raises:
        ModelLoadError: If model file not found or load fails
        SchemaLoadError: If schema validation fails
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ModelLoadError(
            "xgboost is required. Install with: pip install xgboost"
        )
    
    model_path = Path(model_dir)
    model_file, schema_file = resolve_model_paths(
        model_path, model_name, version
    )
    
    if not model_file.exists():
        raise ModelLoadError(f"Model file not found: {model_file}")
    
    schema = load_model_schema(schema_file)
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(model_file))
    except Exception as e:
        raise ModelLoadError(f"Failed to load XGBoost model: {e}")
    
    return model, schema


def get_available_models(model_dir: str) -> Dict[str, list]:
    """List available model versions in a directory.
    
    Scans for files matching pattern: {name}_v{version}.json
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dict mapping model names to list of available versions
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return {}
    
    models = {}
    for f in model_path.glob("*_v*.json"):
        if "_schema_" in f.name:
            continue
        parts = f.stem.split("_v")
        if len(parts) == 2:
            name = parts[0]
            version = parts[1]
            if name not in models:
                models[name] = []
            if version not in models[name]:
                models[name].append(version)
    
    return models
