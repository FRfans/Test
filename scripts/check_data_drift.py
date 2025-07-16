"""
Data Drift Detection Script
Checks for data drift between original and current data using statistical tests
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class DataDriftDetector:
    def __init__(self, reference_data_path="Data/personality_datasert.csv", 
                 current_data_path="Data/synthetic_ctgan_data.csv"):
        self.reference_data_path = reference_data_path
        self.current_data_path = current_data_path
        self.drift_threshold = 0.05  # p-value threshold
        
    def load_data(self):
        """Load reference and current data"""
        print("📂 Loading data for drift detection...")
        
        # Load reference data
        if os.path.exists(self.reference_data_path):
            self.reference_data = pd.read_csv(self.reference_data_path)
            print(f"✅ Reference data loaded: {self.reference_data.shape}")
        else:
            raise FileNotFoundError(f"Reference data not found: {self.reference_data_path}")
        
        # Load current data
        if os.path.exists(self.current_data_path):
            self.current_data = pd.read_csv(self.current_data_path)
            print(f"✅ Current data loaded: {self.current_data.shape}")
        else:
            print(f"⚠️ Current data not found: {self.current_data_path}")
            self.current_data = None
        
        return self.reference_data, self.current_data
    
    def detect_numerical_drift(self, column):
        """Detect drift in numerical columns using Kolmogorov-Smirnov test"""
        if self.current_data is None:
            return None
            
        ref_values = self.reference_data[column].dropna()
        curr_values = self.current_data[column].dropna()
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        
        drift_detected = p_value < self.drift_threshold
        
        return {
            'column': column,
            'test': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': bool(drift_detected),
            'reference_mean': float(ref_values.mean()),
            'current_mean': float(curr_values.mean()) if len(curr_values) > 0 else None,
            'reference_std': float(ref_values.std()),
            'current_std': float(curr_values.std()) if len(curr_values) > 0 else None
        }
    
    def detect_categorical_drift(self, column):
        """Detect drift in categorical columns using Chi-square test"""
        if self.current_data is None:
            return None
            
        ref_values = self.reference_data[column].dropna()
        curr_values = self.current_data[column].dropna()
        
        # Get value counts
        ref_counts = ref_values.value_counts()
        curr_counts = curr_values.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
        
        # Perform Chi-square test
        try:
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
            drift_detected = p_value < self.drift_threshold
        except:
            statistic, p_value = None, None
            drift_detected = False
        
        return {
            'column': column,
            'test': 'Chi-square',
            'statistic': float(statistic) if statistic is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
            'drift_detected': bool(drift_detected),
            'reference_distribution': dict(ref_counts),
            'current_distribution': dict(curr_counts) if len(curr_counts) > 0 else None
        }
    
    def check_all_drifts(self):
        """Check drift for all columns"""
        print("🔍 Checking for data drift...")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'reference_data_path': self.reference_data_path,
            'current_data_path': self.current_data_path,
            'drift_threshold': float(self.drift_threshold),
            'columns_with_drift': [],
            'column_results': {},
            'overall_drift_detected': False
        }
        
        if self.current_data is None:
            drift_results['overall_drift_detected'] = True
            drift_results['reason'] = 'Current data file not found'
            return drift_results
        
        # Check numerical columns
        numerical_columns = self.reference_data.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            if column in self.current_data.columns:
                result = self.detect_numerical_drift(column)
                drift_results['column_results'][column] = result
                
                if result['drift_detected']:
                    drift_results['columns_with_drift'].append(column)
                    print(f"⚠️ Drift detected in {column} (p-value: {result['p_value']:.4f})")
                else:
                    if result['p_value'] is not None:
                        print(f"✅ No drift in {column} (p-value: {result['p_value']:.4f})")
                    else:
                        print(f"✅ No drift in {column} (test skipped)")
        
        # Check categorical columns
        categorical_columns = self.reference_data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column in self.current_data.columns:
                result = self.detect_categorical_drift(column)
                drift_results['column_results'][column] = result
                
                if result['drift_detected']:
                    drift_results['columns_with_drift'].append(column)
                    print(f"⚠️ Drift detected in {column} (p-value: {result['p_value']:.4f})")
                else:
                    if result['p_value'] is not None:
                        print(f"✅ No drift in {column} (p-value: {result['p_value']:.4f})")
                    else:
                        print(f"✅ No drift in {column} (test skipped)")
        
        # Determine overall drift
        drift_results['overall_drift_detected'] = bool(len(drift_results['columns_with_drift']) > 0)
        
        return drift_results
    
    def save_drift_report(self, drift_results, output_path="drift_report.json"):
        """Save drift detection results"""
        with open(output_path, 'w') as f:
            json.dump(drift_results, f, indent=2, cls=NumpyEncoder)
        print(f"📄 Drift report saved to: {output_path}")
    
    def create_drift_summary(self, drift_results):
        """Create a summary of drift detection results"""
        print("\n📊 Drift Detection Summary:")
        print(f"   Timestamp: {drift_results['timestamp']}")
        print(f"   Reference data: {drift_results['reference_data_path']}")
        print(f"   Current data: {drift_results['current_data_path']}")
        print(f"   Drift threshold: {drift_results['drift_threshold']}")
        print(f"   Overall drift detected: {drift_results['overall_drift_detected']}")
        
        if drift_results['columns_with_drift']:
            print(f"   Columns with drift: {drift_results['columns_with_drift']}")
        else:
            print("   No columns with significant drift detected")
        
        return drift_results['overall_drift_detected']

def main():
    """Main drift detection pipeline"""
    print("🎯 Starting data drift detection...")
    
    # Initialize detector
    detector = DataDriftDetector()
    
    # Load data
    try:
        detector.load_data()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        # Create drift status file for CI/CD
        with open("drift_status.txt", "w") as f:
            f.write("true")  # Force retraining if reference data missing
        return
    
    # Check for drift
    drift_results = detector.check_all_drifts()
    
    # Save results
    detector.save_drift_report(drift_results)
    
    # Create summary
    drift_detected = detector.create_drift_summary(drift_results)
    
    # Create drift status file for CI/CD pipeline
    with open("drift_status.txt", "w") as f:
        f.write("true" if drift_detected else "false")
    
    if drift_detected:
        print("\n🚨 Data drift detected! Model retraining recommended.")
    else:
        print("\n✅ No significant data drift detected.")
    
    return drift_detected

if __name__ == "__main__":
    main()
