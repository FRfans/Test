import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Try different Evidently import patterns based on version
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        # Alternative import for older versions
        from evidently.dashboard import Dashboard
        from evidently.dashboard.tabs import DataDriftTab, DataQualityTab
        EVIDENTLY_AVAILABLE = True
        print("⚠️ Using older Evidently API")
    except ImportError:
        print("⚠️ Evidently not properly installed - using fallback")
        EVIDENTLY_AVAILABLE = False

import mlflow
import mlflow.sklearn


class DataDriftDetector:
    def __init__(self, reference_data_path="Data/personality_datasert.csv"):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.current_data = None
        self.drift_report = None
        self.results_dir = "Results"
        
    def load_reference_data(self):
        """Load reference data for drift detection"""
        print("Loading reference data...")
        self.reference_data = pd.read_csv(self.reference_data_path)
        
        # Preprocess reference data
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.reference_data.columns:
                self.reference_data[col] = self.reference_data[col].map({"Yes": 1, "No": 0})
        
        print(f"Reference data shape: {self.reference_data.shape}")
        return self.reference_data
    
    def load_current_data(self, current_data_path="Data/synthetic_ctgan_data.csv"):
        """Load current data for comparison"""
        print("Loading current data...")
        self.current_data = pd.read_csv(current_data_path)
        
        # Preprocess current data
        categorical_columns = ["Stage_fear", "Drained_after_socializing"]
        for col in categorical_columns:
            if col in self.current_data.columns:
                self.current_data[col] = self.current_data[col].map({"Yes": 1, "No": 0})
        
        print(f"Current data shape: {self.current_data.shape}")
        return self.current_data
    
    def detect_drift(self):
        """Detect data drift between reference and current data"""
        if not EVIDENTLY_AVAILABLE:
            print("⚠️ Evidently not available - using fallback drift detection")
            return self._fallback_drift_detection()
        
        if self.reference_data is None:
            self.load_reference_data()
        if self.current_data is None:
            self.load_current_data()
        
        print("Detecting data drift with Evidently...")
        
        try:
            # Ensure both datasets have the same columns
            common_columns = set(self.reference_data.columns).intersection(set(self.current_data.columns))
            common_columns.discard('Personality')  # Remove target column
            
            ref_data = self.reference_data[list(common_columns)]
            curr_data = self.current_data[list(common_columns)]
            
            # Create data drift report
            data_drift_report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])
            
            data_drift_report.run(reference_data=ref_data, current_data=curr_data)
            
            # Save HTML report
            os.makedirs(self.results_dir, exist_ok=True)
            report_path = os.path.join(self.results_dir, "data_drift_report.html")
            data_drift_report.save_html(report_path)
            print(f"✓ Drift report saved to: {report_path}")
            
            # Extract drift metrics
            drift_results = data_drift_report.as_dict()
            
            # Save JSON results
            json_path = os.path.join(self.results_dir, "drift_results.json")
            with open(json_path, 'w') as f:
                json.dump(drift_results, f, indent=2)
            
            # Extract key metrics
            drift_metrics = self._extract_drift_metrics(drift_results)
            
            # Save summary
            summary_path = os.path.join(self.results_dir, "drift_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Data Drift Detection Summary\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Reference data shape: {ref_data.shape}\n")
                f.write(f"Current data shape: {curr_data.shape}\n")
                f.write(f"Number of drifted features: {drift_metrics['n_drifted_features']}\n")
                f.write(f"Share of drifted features: {drift_metrics['share_drifted_features']:.2%}\n")
                f.write(f"Dataset drift detected: {drift_metrics['dataset_drift']}\n")
            
            print(f"✓ Drift results saved to: {json_path}")
            print(f"✓ Drift summary saved to: {summary_path}")
            
            return drift_metrics
            
        except Exception as e:
            print(f"⚠️ Evidently drift detection failed: {e}")
            return self._fallback_drift_detection()
    
    def _fallback_drift_detection(self):
        """Simple fallback drift detection without Evidently"""
        if self.reference_data is None:
            self.load_reference_data()
        if self.current_data is None:
            self.load_current_data()
        
        print("Running simple statistical drift detection...")
        
        # Simple statistical comparison
        common_columns = set(self.reference_data.columns).intersection(set(self.current_data.columns))
        common_columns.discard('Personality')
        
        ref_data = self.reference_data[list(common_columns)]
        curr_data = self.current_data[list(common_columns)]
        
        drifted_features = 0
        total_features = len(common_columns)
        
        # Simple statistical tests
        for col in common_columns:
            if col in ref_data.columns and col in curr_data.columns:
                # Compare means (simple test)
                ref_mean = ref_data[col].mean()
                curr_mean = curr_data[col].mean()
                
                # Simple threshold-based drift detection
                if abs(ref_mean - curr_mean) > 0.1 * abs(ref_mean):
                    drifted_features += 1
        
        drift_metrics = {
            'n_drifted_features': drifted_features,
            'share_drifted_features': drifted_features / total_features if total_features > 0 else 0.0,
            'dataset_drift': drifted_features > total_features * 0.3  # 30% threshold
        }
        
        # Save summary
        os.makedirs(self.results_dir, exist_ok=True)
        summary_path = os.path.join(self.results_dir, "drift_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Simple Data Drift Detection Summary\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Reference data shape: {ref_data.shape}\n")
            f.write(f"Current data shape: {curr_data.shape}\n")
            f.write(f"Number of drifted features: {drift_metrics['n_drifted_features']}\n")
            f.write(f"Share of drifted features: {drift_metrics['share_drifted_features']:.2%}\n")
            f.write(f"Dataset drift detected: {drift_metrics['dataset_drift']}\n")
            f.write("Note: Simple statistical drift detection used\n")
        
        print(f"✓ Simple drift summary saved to: {summary_path}")
        
        return drift_metrics
    
    def _extract_drift_metrics(self, drift_results):
        """Extract key metrics from drift results"""
        try:
            metrics = drift_results['metrics'][0]['result']
            return {
                'n_drifted_features': metrics.get('number_of_drifted_columns', 0),
                'share_drifted_features': metrics.get('share_of_drifted_columns', 0.0),
                'dataset_drift': metrics.get('dataset_drift', False)
            }
        except (KeyError, IndexError):
            return {
                'n_drifted_features': 0,
                'share_drifted_features': 0.0,
                'dataset_drift': False
            }
    
    def run_drift_tests(self):
        """Run drift tests with pass/fail results"""
        if not EVIDENTLY_AVAILABLE:
            print("⚠️ Evidently not available - using simple drift tests")
            return self._fallback_drift_tests()
            
        if self.reference_data is None:
            self.load_reference_data()
        if self.current_data is None:
            self.load_current_data()
        
        print("Running Evidently drift tests...")
        
        try:
            # Ensure both datasets have the same columns
            common_columns = set(self.reference_data.columns).intersection(set(self.current_data.columns))
            common_columns.discard('Personality')  # Remove target column
            
            ref_data = self.reference_data[list(common_columns)]
            curr_data = self.current_data[list(common_columns)]
            
            # Create test suite
            data_drift_test_suite = TestSuite(tests=[
                DataDriftTestPreset()
            ])
            
            data_drift_test_suite.run(reference_data=ref_data, current_data=curr_data)
            
            # Save test results
            test_report_path = os.path.join(self.results_dir, "drift_test_results.html")
            data_drift_test_suite.save_html(test_report_path)
            
            # Get test results
            test_results = data_drift_test_suite.as_dict()
            
            # Count passed/failed tests
            tests = test_results.get('tests', [])
            passed_tests = sum(1 for test in tests if test.get('status') == 'SUCCESS')
            total_tests = len(tests)
            
            print(f"✓ Drift tests completed: {passed_tests}/{total_tests} passed")
            print(f"✓ Test results saved to: {test_report_path}")
            
            return {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'test_success_rate': passed_tests / total_tests if total_tests > 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️ Evidently tests failed: {e}")
            return self._fallback_drift_tests()
    
    def _fallback_drift_tests(self):
        """Simple fallback drift tests"""
        # Run simple drift detection first
        drift_metrics = self._fallback_drift_detection()
        
        # Create simple test results
        dataset_drift_test = 1 if not drift_metrics['dataset_drift'] else 0  # Pass if no drift
        feature_drift_test = 1 if drift_metrics['share_drifted_features'] < 0.3 else 0  # Pass if <30% drift
        
        total_tests = 2
        passed_tests = dataset_drift_test + feature_drift_test
        
        print(f"✓ Simple drift tests completed: {passed_tests}/{total_tests} passed")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'test_success_rate': passed_tests / total_tests
        }


def main():
    """Main function to run data drift detection"""
    detector = DataDriftDetector()
    
    # Run drift detection
    drift_metrics = detector.detect_drift()
    test_results = detector.run_drift_tests()
    
    # Log to MLflow if available
    try:
        with mlflow.start_run(run_name="Data_Drift_Detection"):
            mlflow.log_metrics(drift_metrics)
            mlflow.log_metrics(test_results)
            mlflow.log_artifact("Results/data_drift_report.html")
            mlflow.log_artifact("Results/drift_results.json")
            mlflow.log_artifact("Results/drift_summary.txt")
            mlflow.log_artifact("Results/drift_test_results.html")
        print("✓ Drift metrics logged to MLflow")
    except Exception as e:
        print(f"⚠️ Could not log to MLflow: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA DRIFT DETECTION SUMMARY")
    print("="*60)
    print(f"Number of drifted features: {drift_metrics['n_drifted_features']}")
    print(f"Share of drifted features: {drift_metrics['share_drifted_features']:.2%}")
    print(f"Dataset drift detected: {drift_metrics['dataset_drift']}")
    print(f"Drift tests passed: {test_results['passed_tests']}/{test_results['total_tests']}")
    print("="*60)
    
    return drift_metrics, test_results


if __name__ == "__main__":
    drift_metrics, test_results = main()