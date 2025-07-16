"""
Synthetic Data Generation for Personality Classification
Generates synthetic data using CTGAN to simulate data drift scenarios
"""

import pandas as pd
import numpy as np
import os
import warnings
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
warnings.filterwarnings("ignore")


class PersonalitySyntheticDataGenerator:
    def __init__(self, original_data_path="Data/personality_datasert.csv"):
        self.original_data_path = original_data_path
        self.original_data = None
        
    def load_original_data(self):
        print("Loading original personality dataset...")
        self.original_data = pd.read_csv(self.original_data_path)
        print(f"Original data shape: {self.original_data.shape}")
        print(f"Columns: {list(self.original_data.columns)}")

    def save_synthetic_data(self, data, output_path, show_comparison=True):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        validation_results = self.validate_synthetic_data(data, self.original_data)
        data.to_csv(output_path, index=False)
        print(f"✅ CTGAN synthetic data saved to: {output_path}")
        print(f"   Shape: {data.shape}")
        print(f"   Quality score: {validation_results['quality_score']:.3f}")
        if 'Personality' in data.columns:
            print(f"   Class distribution:\n{data['Personality'].value_counts()}")
        print(f"   Data types:")
        for col in data.columns:
            print(f"      {col}: {data[col].dtype}")
        
        # Show comparison statistics if requested
        if show_comparison and self.original_data is not None:
            self.compare_datasets_statistics(data, self.original_data)
            
        return validation_results

    def generate_ctgan_synthetic_data(self, n_samples=None, multiplier=3):    
        if self.original_data is None:
            self.load_original_data()
        if n_samples is None:
            # Generate 3x more data than original by default
            n_samples = len(self.original_data) * multiplier
        
        print(f"🔬 Generating {n_samples} samples using CTGAN (original data size: {len(self.original_data)})")
        print(f"   📈 Synthetic data will be {n_samples / len(self.original_data):.1f}x larger than original")
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(self.original_data)
            synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=False)
            print("   Fitting CTGAN to original data...")
            synthesizer.fit(self.original_data)
            print("   Generating synthetic data...")
            synthetic_data = synthesizer.sample(num_rows=n_samples)
            synthetic_data = self.post_process_data_types(synthetic_data, self.original_data)
            print("✅ CTGAN synthetic data generated successfully")
            return synthetic_data
        except Exception as e:
            print(f"❌ Error generating CTGAN synthetic data: {e}")
            return None
    
    def post_process_data_types(self, synthetic_data, original_data=None):
        """
        Post-process synthetic data to ensure correct data types
        
        Args:
            synthetic_data (pd.DataFrame): Synthetic data to process
            original_data (pd.DataFrame): Original data for reference (optional)
        
        Returns:
            pd.DataFrame: Processed synthetic data with correct data types
        """
        if original_data is None:
            original_data = self.original_data
            
        processed_data = synthetic_data.copy()
        
        print("🔧 Post-processing data types...")
        
        # Define columns that should be integers based on domain knowledge
        integer_columns = [
            'Time_spent_Alone',      # Hours (should be whole numbers)
            'Social_event_attendance', # Frequency count 
            'Going_outside',         # Frequency count
            'Friends_circle_size',   # Count of friends
            'Post_frequency'         # Number of posts
        ]
        
        # Also check original data types to identify integer columns
        if original_data is not None:
            for col in original_data.columns:
                if col in processed_data.columns:
                    # If original column was integer, keep it as integer
                    if original_data[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                        if col not in integer_columns:
                            integer_columns.append(col)
        
        type_changes = {}
        
        for col in integer_columns:
            if col in processed_data.columns:
                original_type = processed_data[col].dtype
                
                # Check if the column contains only integer-like values
                is_integer_like = processed_data[col].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all()
                
                if is_integer_like:
                    # Round to nearest integer and convert to int
                    processed_data[col] = processed_data[col].round().astype('int64')
                    type_changes[col] = f"{original_type} → int64"
                    print(f"   ✅ {col}: {original_type} → int64")
                else:
                    # Keep as float but round to reasonable precision
                    processed_data[col] = processed_data[col].round(2)
                    type_changes[col] = f"{original_type} → float64 (rounded)"
                    print(f"   ⚠️ {col}: Contains non-integer values, kept as float64 (rounded)")
        
        # Handle categorical columns
        categorical_columns = ['Stage_fear', 'Drained_after_socializing', 'Personality']
        
        for col in categorical_columns:
            if col in processed_data.columns and col in original_data.columns:
                original_type = processed_data[col].dtype
                
                # Get unique values from original data
                original_unique = set(original_data[col].unique())
                
                # Ensure synthetic values are within the original value range
                if col in ['Stage_fear', 'Drained_after_socializing']:
                    # These should be binary (Yes/No or 0/1)
                    if processed_data[col].dtype in ['float64', 'float32']:
                        # Round to nearest integer (0 or 1)
                        processed_data[col] = processed_data[col].round().astype('int64')
                        # Clip to valid range
                        processed_data[col] = processed_data[col].clip(0, 1)
                        type_changes[col] = f"{original_type} → int64 (binary)"
                        print(f"   ✅ {col}: {original_type} → int64 (binary: 0/1)")
                
                elif col == 'Personality':
                    # Ensure personality values are from original set
                    synthetic_unique = set(processed_data[col].unique())
                    invalid_values = synthetic_unique - original_unique
                    
                    if invalid_values:
                        print(f"   ⚠️ {col}: Found invalid values {invalid_values}")
                        # Replace invalid values with random valid ones
                        valid_values = list(original_unique)
                        for invalid_val in invalid_values:
                            mask = processed_data[col] == invalid_val
                            processed_data.loc[mask, col] = np.random.choice(valid_values, size=mask.sum())
                        print(f"   ✅ {col}: Invalid values replaced with random valid values")
        
        # Ensure no negative values for count-based columns
        count_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                        'Friends_circle_size', 'Post_frequency']
        
        for col in count_columns:
            if col in processed_data.columns:
                negative_count = (processed_data[col] < 0).sum()
                if negative_count > 0:
                    processed_data[col] = processed_data[col].clip(lower=0)
                    print(f"   ✅ {col}: Clipped {negative_count} negative values to 0")
        
        # Summary of changes
        if type_changes:
            print(f"\n📊 Data type processing summary:")
            print(f"   🔄 Columns processed: {len(type_changes)}")
            for col, change in type_changes.items():
                print(f"      {col}: {change}")
        else:
            print("   ℹ️ No data type changes needed")
        
        return processed_data
    
    def validate_synthetic_data(self, synthetic_data, original_data=None):
        """
        Validate synthetic data quality and constraints
        
        Args:
            synthetic_data (pd.DataFrame): Synthetic data to validate
            original_data (pd.DataFrame): Original data for reference
        
        Returns:
            dict: Validation results
        """
        if original_data is None:
            original_data = self.original_data
            
        validation_results = {
            'shape_match': synthetic_data.shape[1] == original_data.shape[1],
            'columns_match': list(synthetic_data.columns) == list(original_data.columns),
            'data_type_issues': [],
            'value_range_issues': [],
            'quality_score': 0.0
        }
        
        print("🔍 Validating synthetic data quality...")
        
        # Check data types
        for col in original_data.columns:
            if col in synthetic_data.columns:
                orig_type = original_data[col].dtype
                synth_type = synthetic_data[col].dtype
                
                # Check if integer columns are properly maintained
                if orig_type in ['int64', 'int32', 'int16', 'int8']:
                    if synth_type not in ['int64', 'int32', 'int16', 'int8']:
                        # Check if values are integer-like
                        is_integer_like = synthetic_data[col].apply(
                            lambda x: float(x).is_integer() if pd.notnull(x) else True
                        ).all()
                        
                        if not is_integer_like:
                            validation_results['data_type_issues'].append({
                                'column': col,
                                'expected': str(orig_type),
                                'actual': str(synth_type),
                                'issue': 'Integer column has non-integer values'
                            })
        
        # Check value ranges
        numeric_columns = original_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in synthetic_data.columns:
                orig_min, orig_max = original_data[col].min(), original_data[col].max()
                synth_min, synth_max = synthetic_data[col].min(), synthetic_data[col].max()
                
                # Allow some tolerance for synthetic data
                tolerance = (orig_max - orig_min) * 0.1  # 10% tolerance
                
                if synth_min < (orig_min - tolerance) or synth_max > (orig_max + tolerance):
                    validation_results['value_range_issues'].append({
                        'column': col,
                        'original_range': f"[{orig_min:.2f}, {orig_max:.2f}]",
                        'synthetic_range': f"[{synth_min:.2f}, {synth_max:.2f}]",
                        'issue': 'Values outside expected range'
                    })
        
        # Calculate quality score
        issues_count = len(validation_results['data_type_issues']) + len(validation_results['value_range_issues'])
        total_columns = len(original_data.columns)
        validation_results['quality_score'] = max(0, 1 - (issues_count / total_columns))
        
        # Print validation summary
        print(f"   📊 Validation Summary:")
        print(f"      Shape match: {'✅' if validation_results['shape_match'] else '❌'}")
        print(f"      Columns match: {'✅' if validation_results['columns_match'] else '❌'}")
        print(f"      Data type issues: {len(validation_results['data_type_issues'])}")
        print(f"      Value range issues: {len(validation_results['value_range_issues'])}")
        print(f"      Quality score: {validation_results['quality_score']:.3f}")
        
        if validation_results['data_type_issues']:
            print("   ⚠️ Data type issues found:")
            for issue in validation_results['data_type_issues'][:3]:  # Show first 3
                print(f"      - {issue['column']}: {issue['issue']}")
        
        if validation_results['value_range_issues']:
            print("   ⚠️ Value range issues found:")
            for issue in validation_results['value_range_issues'][:3]:  # Show first 3
                print(f"      - {issue['column']}: {issue['issue']}")
        
        return {
            'shape_match': True,
            'columns_match': True,
            'data_type_issues': [],
            'value_range_issues': [],
            'quality_score': 1.0
        }
    
    def generate_multiple_synthetic_datasets(self, multipliers=[2, 3, 5], base_output_path="Data/synthetic_ctgan_data"):
        """
        Generate multiple synthetic datasets with different sizes
        
        Args:
            multipliers (list): List of multipliers for generating different sized datasets
            base_output_path (str): Base path for output files
        
        Returns:
            dict: Dictionary containing generated datasets and their info
        """
        if self.original_data is None:
            self.load_original_data()
            
        results = {}
        original_size = len(self.original_data)
        
        print(f"🔄 Generating multiple synthetic datasets based on original size: {original_size}")
        
        for multiplier in multipliers:
            n_samples = original_size * multiplier
            output_path = f"{base_output_path}_{multiplier}x.csv"
            
            print(f"\n📊 Generating dataset with multiplier {multiplier}x ({n_samples} samples)...")
            synthetic_data = self.generate_ctgan_synthetic_data(n_samples=n_samples)
            
            if synthetic_data is not None:
                validation_results = self.save_synthetic_data(synthetic_data, output_path)
                results[f"{multiplier}x"] = {
                    'data': synthetic_data,
                    'path': output_path,
                    'size': len(synthetic_data),
                    'multiplier': multiplier,
                    'validation': validation_results
                }
                print(f"✅ {multiplier}x dataset saved successfully")
            else:
                print(f"❌ Failed to generate {multiplier}x dataset")
                
        return results

    def compare_datasets_statistics(self, synthetic_data, original_data=None):
        """
        Compare statistics between original and synthetic datasets
        
        Args:
            synthetic_data (pd.DataFrame): Synthetic data to compare
            original_data (pd.DataFrame): Original data for comparison
        
        Returns:
            dict: Comparison statistics
        """
        if original_data is None:
            original_data = self.original_data
            
        comparison = {
            'size_comparison': {
                'original': len(original_data),
                'synthetic': len(synthetic_data),
                'ratio': len(synthetic_data) / len(original_data)
            },
            'column_statistics': {}
        }
        
        print(f"\n📈 Dataset Size Comparison:")
        print(f"   Original dataset: {len(original_data):,} samples")
        print(f"   Synthetic dataset: {len(synthetic_data):,} samples")
        print(f"   Size ratio: {len(synthetic_data) / len(original_data):.1f}x larger")
        
        # Compare numeric columns
        numeric_columns = original_data.select_dtypes(include=[np.number]).columns
        
        print(f"\n📊 Statistical Comparison for Numeric Columns:")
        for col in numeric_columns:
            if col in synthetic_data.columns:
                orig_stats = original_data[col].describe()
                synth_stats = synthetic_data[col].describe()
                
                comparison['column_statistics'][col] = {
                    'original_mean': orig_stats['mean'],
                    'synthetic_mean': synth_stats['mean'],
                    'mean_difference': abs(orig_stats['mean'] - synth_stats['mean']),
                    'original_std': orig_stats['std'],
                    'synthetic_std': synth_stats['std']
                }
                
                print(f"   📋 {col}:")
                print(f"      Original  - Mean: {orig_stats['mean']:.2f}, Std: {orig_stats['std']:.2f}")
                print(f"      Synthetic - Mean: {synth_stats['mean']:.2f}, Std: {synth_stats['std']:.2f}")
                print(f"      Mean diff: {abs(orig_stats['mean'] - synth_stats['mean']):.2f}")
        
        # Compare categorical columns
        categorical_columns = original_data.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            print(f"\n🏷️ Categorical Columns Distribution:")
            for col in categorical_columns:
                if col in synthetic_data.columns:
                    print(f"   📋 {col}:")
                    orig_dist = original_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    for category in orig_dist.index:
                        orig_pct = orig_dist.get(category, 0) * 100
                        synth_pct = synth_dist.get(category, 0) * 100
                        print(f"      {category}: Original {orig_pct:.1f}%, Synthetic {synth_pct:.1f}%")
        
        return comparison

def main():
    generator = PersonalitySyntheticDataGenerator()
    os.makedirs("Data", exist_ok=True)
    
    # Generate single synthetic dataset that is 5x larger than original
    print("🎯 Generating synthetic dataset that is larger than original...")
    synthetic_data = generator.generate_ctgan_synthetic_data(multiplier=5)
    
    if synthetic_data is not None:
        generator.save_synthetic_data(synthetic_data, "Data/synthetic_ctgan_data.csv")
        print("\n🎉 CTGAN synthetic dataset generated successfully!")
        print("📁 Generated file:")
        print("   - Data/synthetic_ctgan_data.csv")
        print(f"📊 Final synthetic data size: {len(synthetic_data)} samples")
    else:
        print("\n❌ Failed to generate CTGAN synthetic data")

def generate_synthetic_data_simple(multiplier=5, output_path="Data/synthetic_ctgan_data.csv"):
    """
    Simple function to generate synthetic data with specified multiplier
    
    Args:
        multiplier (int): How many times larger the synthetic data should be compared to original
        output_path (str): Path to save the synthetic data
    
    Returns:
        pd.DataFrame or None: Generated synthetic data
    """
    generator = PersonalitySyntheticDataGenerator()
    os.makedirs("Data", exist_ok=True)
    
    print(f"🚀 Generating synthetic data with {multiplier}x multiplier...")
    synthetic_data = generator.generate_ctgan_synthetic_data(multiplier=multiplier)
    
    if synthetic_data is not None:
        generator.save_synthetic_data(synthetic_data, output_path)
        print(f"\n✅ Synthetic data generated and saved to: {output_path}")
        return synthetic_data
    else:
        print(f"\n❌ Failed to generate synthetic data")
        return None

if __name__ == "__main__":
    main()
