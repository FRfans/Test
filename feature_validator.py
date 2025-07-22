"""
Feature validation utility for Personality Classifier project.
Ensures consistency between training data, model, and prediction inputs.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class FeatureValidator:
    """Validates feature consistency across the ML pipeline"""
    
    # Define the canonical feature schema for the project
    CANONICAL_FEATURES = [
        'Time_spent_Alone',
        'Stage_fear', 
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ]
    
    CATEGORICAL_FEATURES = ['Stage_fear', 'Drained_after_socializing']
    TARGET_FEATURE = 'Personality'
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, stage: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Validate a dataframe has the expected features
        
        Args:
            df: DataFrame to validate
            stage: Stage of pipeline (training, prediction, etc.)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for target column in training data
        if stage == "training":
            if cls.TARGET_FEATURE not in df.columns:
                issues.append(f"Missing target feature: {cls.TARGET_FEATURE}")
                
        # Check required features
        df_features = set(df.columns) - {cls.TARGET_FEATURE}
        expected_features = set(cls.CANONICAL_FEATURES)
        
        missing_features = expected_features - df_features
        extra_features = df_features - expected_features
        
        if missing_features:
            issues.append(f"Missing features: {list(missing_features)}")
            
        if extra_features:
            issues.append(f"Unexpected features: {list(extra_features)}")
            
        # Check feature types
        for feature in cls.CATEGORICAL_FEATURES:
            if feature in df.columns:
                unique_values = set(df[feature].dropna().unique())
                expected_values = {0, 1, "Yes", "No"}
                if not unique_values.issubset(expected_values):
                    invalid_values = unique_values - expected_values
                    issues.append(f"Invalid values in {feature}: {invalid_values}")
        
        return len(issues) == 0, issues
    
    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize a dataframe to match expected format
        
        Args:
            df: Input dataframe
            
        Returns:
            Standardized dataframe
        """
        df_copy = df.copy()
        
        # Convert categorical features to numeric if needed
        for col in cls.CATEGORICAL_FEATURES:
            if col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].map({"Yes": 1, "No": 0})
        
        # Ensure feature order
        available_features = [f for f in cls.CANONICAL_FEATURES if f in df_copy.columns]
        if cls.TARGET_FEATURE in df_copy.columns:
            available_features.append(cls.TARGET_FEATURE)
            
        return df_copy[available_features]
    
    @classmethod
    def save_feature_schema(cls, output_dir: str = "Model"):
        """Save the canonical feature schema to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        schema = {
            "canonical_features": cls.CANONICAL_FEATURES,
            "categorical_features": cls.CATEGORICAL_FEATURES,
            "numerical_features": [f for f in cls.CANONICAL_FEATURES if f not in cls.CATEGORICAL_FEATURES],
            "target_feature": cls.TARGET_FEATURE
        }
        
        # Save as JSON
        with open(os.path.join(output_dir, "feature_schema.json"), "w") as f:
            json.dump(schema, f, indent=2)
            
        # Save just feature names for backward compatibility
        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(cls.CANONICAL_FEATURES, f, indent=2)
            
        print(f"✅ Feature schema saved to {output_dir}/")
        
    @classmethod
    def load_feature_schema(cls, model_dir: str = "Model") -> Dict[str, Any]:
        """Load feature schema from files"""
        schema_file = os.path.join(model_dir, "feature_schema.json")
        
        if os.path.exists(schema_file):
            with open(schema_file, "r") as f:
                return json.load(f)
        else:
            # Fallback to canonical schema
            return {
                "canonical_features": cls.CANONICAL_FEATURES,
                "categorical_features": cls.CATEGORICAL_FEATURES,
                "numerical_features": [f for f in cls.CANONICAL_FEATURES if f not in cls.CATEGORICAL_FEATURES],
                "target_feature": cls.TARGET_FEATURE
            }
    
    @classmethod
    def create_sample_data(cls, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Create sample data with correct feature structure for CI/CD
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with sample data
        """
        np.random.seed(random_state)
        
        sample_data = {
            'Time_spent_Alone': np.random.randint(0, 12, n_samples),
            'Stage_fear': np.random.choice(['Yes', 'No'], n_samples),
            'Social_event_attendance': np.random.randint(0, 10, n_samples),
            'Going_outside': np.random.randint(0, 10, n_samples),
            'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples),
            'Friends_circle_size': np.random.randint(0, 20, n_samples),
            'Post_frequency': np.random.randint(0, 10, n_samples),
            'Personality': np.random.choice(['Introvert', 'Extrovert'], n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Validate the created data
        is_valid, issues = cls.validate_dataframe(df, stage="training")
        if not is_valid:
            raise ValueError(f"Generated sample data is invalid: {issues}")
            
        return df
    
    @classmethod
    def validate_prediction_input(cls, input_data: Dict[str, Any]) -> Tuple[bool, List[str], List[float]]:
        """
        Validate and prepare prediction input
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Tuple of (is_valid, issues, feature_array)
        """
        issues = []
        
        # Check all required features are present
        missing_features = set(cls.CANONICAL_FEATURES) - set(input_data.keys())
        if missing_features:
            issues.append(f"Missing features: {list(missing_features)}")
            return False, issues, []
        
        # Validate categorical features
        for feature in cls.CATEGORICAL_FEATURES:
            if feature in input_data:
                value = input_data[feature]
                if value not in [0, 1, "Yes", "No"]:
                    issues.append(f"Invalid value for {feature}: {value}")
        
        # Build feature array in correct order
        try:
            feature_array = []
            for feature in cls.CANONICAL_FEATURES:
                value = input_data[feature]
                
                # Convert categorical to numeric if needed
                if feature in cls.CATEGORICAL_FEATURES and isinstance(value, str):
                    value = 1 if value == "Yes" else 0
                    
                feature_array.append(float(value))
                
        except (ValueError, KeyError) as e:
            issues.append(f"Error processing features: {e}")
            return False, issues, []
        
        return len(issues) == 0, issues, feature_array


def main():
    """Test the feature validator"""
    print("🧪 Testing Feature Validator...")
    
    # Create sample data
    print("\n1. Creating sample data...")
    sample_df = FeatureValidator.create_sample_data(100)
    print(f"✅ Created sample data: {sample_df.shape}")
    print(f"   Features: {list(sample_df.columns[:-1])}")
    
    # Validate the sample data
    print("\n2. Validating sample data...")
    is_valid, issues = FeatureValidator.validate_dataframe(sample_df, "training")
    if is_valid:
        print("✅ Sample data is valid")
    else:
        print(f"❌ Sample data issues: {issues}")
    
    # Save feature schema
    print("\n3. Saving feature schema...")
    FeatureValidator.save_feature_schema()
    
    # Test prediction input validation
    print("\n4. Testing prediction input validation...")
    test_input = {
        'Time_spent_Alone': 4.0,
        'Stage_fear': 0,
        'Social_event_attendance': 4.0,
        'Going_outside': 6.0,
        'Drained_after_socializing': 0,
        'Friends_circle_size': 13.0,
        'Post_frequency': 5.0
    }
    
    is_valid, issues, feature_array = FeatureValidator.validate_prediction_input(test_input)
    if is_valid:
        print(f"✅ Prediction input is valid: {len(feature_array)} features")
        print(f"   Feature array: {feature_array}")
    else:
        print(f"❌ Prediction input issues: {issues}")


if __name__ == "__main__":
    main()
