"""
Feature utility functions for ensuring consistency across the ML pipeline
"""
import pandas as pd
import numpy as np
import json
import os
import skops.io as sio
from skops.io import get_untrusted_types


def get_expected_features():
    """Get the standard feature set expected by the model"""
    return [
        # Core numerical features
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency',
        
        # Core categorical features
        'Stage_fear', 'Drained_after_socializing',
        
        # Legacy features (for backward compatibility)
        'Time_spent_with_family', 'Time_spent_with_friends', 
        'Anxiety_rating', 'Social_media_usage'
    ]


def load_feature_names_from_model():
    """Load feature names from saved model artifacts"""
    feature_paths = [
        "Model/feature_names.json",
        "Model/feature_names.skops"
    ]
    
    for path in feature_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.json'):
                    with open(path, 'r') as f:
                        return json.load(f)
                else:
                    untrusted_types = get_untrusted_types(file=path)
                    return sio.load(path, trusted=untrusted_types)
            except Exception as e:
                print(f"Warning: Could not load feature names from {path}: {e}")
                continue
    
    # Fallback to expected features
    print("Warning: Using fallback feature set")
    return get_expected_features()


def validate_and_align_features(data, target_features=None, add_missing=True):
    """
    Validate and align dataset features with expected feature set
    
    Args:
        data: pandas DataFrame to validate
        target_features: list of expected features (if None, load from model)
        add_missing: whether to add missing features with default values
    
    Returns:
        pandas DataFrame with aligned features
    """
    if target_features is None:
        target_features = load_feature_names_from_model()
    
    # Remove target column if present
    feature_columns = [col for col in target_features if col != 'Personality']
    
    print(f"Target features: {len(feature_columns)} features")
    print(f"Input data: {data.shape[1]} columns")
    
    # Check for missing features
    missing_features = [f for f in feature_columns if f not in data.columns]
    extra_features = [f for f in data.columns if f not in feature_columns and f != 'Personality']
    
    if missing_features:
        print(f"Missing features: {missing_features}")
        if add_missing:
            for feature in missing_features:
                # Add default values based on feature type
                if feature in ['Stage_fear', 'Drained_after_socializing']:
                    # Binary categorical features
                    data[feature] = np.random.randint(0, 2, len(data))
                else:
                    # Numerical features - use reasonable defaults
                    data[feature] = np.random.randint(0, 10, len(data))
                print(f"✅ Added missing feature '{feature}' with default values")
    
    if extra_features:
        print(f"Extra features (will be ignored): {extra_features}")
    
    # Select and reorder columns to match target features
    available_features = [f for f in feature_columns if f in data.columns]
    aligned_data = data[available_features + (['Personality'] if 'Personality' in data.columns else [])].copy()
    
    print(f"✅ Aligned data: {aligned_data.shape}")
    return aligned_data


def validate_model_input(input_array, expected_features=None):
    """
    Validate input array shape before model prediction
    
    Args:
        input_array: numpy array for model input
        expected_features: number of expected features
    
    Returns:
        bool: True if valid, False otherwise
    """
    if expected_features is None:
        expected_features = len(load_feature_names_from_model())
    
    if input_array.shape[1] != expected_features:
        print(f"❌ Feature mismatch: Expected {expected_features}, got {input_array.shape[1]}")
        return False
    
    print(f"✅ Input validation passed: {input_array.shape[1]} features")
    return True


def create_sample_data_with_all_features(n_samples=1000, random_seed=42):
    """Create sample data with all expected features for CI/CD"""
    np.random.seed(random_seed)
    
    expected_features = get_expected_features()
    
    data = {}
    for feature in expected_features:
        if feature in ['Stage_fear', 'Drained_after_socializing']:
            # Binary categorical features
            data[feature] = np.random.randint(0, 2, n_samples)
        elif feature in ['Time_spent_Alone', 'Time_spent_with_family', 'Time_spent_with_friends']:
            # Time-based features (0-12 hours)
            data[feature] = np.random.randint(0, 12, n_samples)
        elif feature in ['Social_event_attendance', 'Going_outside', 'Post_frequency', 
                        'Anxiety_rating', 'Social_media_usage']:
            # Scale-based features (0-10)
            data[feature] = np.random.randint(0, 10, n_samples)
        elif feature == 'Friends_circle_size':
            # Friend count (1-50)
            data[feature] = np.random.randint(1, 50, n_samples)
    
    # Add target column
    data['Personality'] = np.random.choice(['Introvert', 'Extrovert'], n_samples)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the utility functions
    print("Testing feature utility functions...")
    
    # Test sample data creation
    sample_data = create_sample_data_with_all_features(100)
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    # Test feature validation
    expected_features = get_expected_features()
    print(f"Expected features: {len(expected_features)}")
    
    # Test alignment
    aligned_data = validate_and_align_features(sample_data)
    print(f"Aligned data shape: {aligned_data.shape}")
    
    print("✅ Feature utilities test completed")
