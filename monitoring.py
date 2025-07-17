import time
import psutil
import os
import json
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import threading
import mlflow
import mlflow.sklearn
from data_drift import DataDriftDetector  # Re-enabled


class MLOpsMonitoring:
    def __init__(self, port=8000):
        self.port = port
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        
        # ML metrics
        self.model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
        self.model_auc = Gauge('model_auc', 'Current model AUC score')
        self.training_duration = Histogram('model_training_duration_seconds', 'Model training duration')
        
        # Data drift metrics
        self.drift_features_count = Gauge('data_drift_features_count', 'Number of features with drift')
        self.drift_share = Gauge('data_drift_share_percent', 'Percentage of features with drift')
        self.dataset_drift = Gauge('dataset_drift_detected', 'Whether dataset drift is detected (1=yes, 0=no)')
        
        # Request metrics
        self.prediction_counter = Counter('predictions_total', 'Total number of predictions made')
        self.prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
        
        # Model info
        self.model_info = Info('model_info', 'Information about the current model')
        
        # Monitoring status
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start Prometheus metrics server"""
        print(f"Starting Prometheus metrics server on port {self.port}...")
        start_http_server(self.port)
        self.monitoring_active = True
        
        # Start background monitoring
        self._start_background_monitoring()
        print(f"✓ Prometheus metrics available at http://localhost:{self.port}/metrics")
        
    def _start_background_monitoring(self):
        """Start background thread for system monitoring"""
        def monitor_system():
            while self.monitoring_active:
                self._update_system_metrics()
                time.sleep(10)  # Update every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        
    def _update_system_metrics(self):
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def update_model_metrics(self, accuracy=None, auc=None, training_time=None):
        """Update model performance metrics"""
        if accuracy is not None:
            self.model_accuracy.set(accuracy)
        if auc is not None:
            self.model_auc.set(auc)
        if training_time is not None:
            self.training_duration.observe(training_time)
    
    def update_drift_metrics(self, drift_metrics):
        """Update data drift metrics"""
        self.drift_features_count.set(drift_metrics.get('n_drifted_features', 0))
        self.drift_share.set(drift_metrics.get('share_drifted_features', 0.0) * 100)
        self.dataset_drift.set(1 if drift_metrics.get('dataset_drift', False) else 0)
    
    def record_prediction(self, latency=None):
        """Record a prediction event"""
        self.prediction_counter.inc()
        if latency is not None:
            self.prediction_latency.observe(latency)
    
    def set_model_info(self, model_name, model_version, training_date):
        """Set model information"""
        self.model_info.info({
            'model_name': model_name,
            'model_version': model_version,
            'training_date': training_date
        })
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        print("\n" + "="*60)
        print("RUNNING MLOPS MONITORING CYCLE")
        print("="*60)
        
        # 1. Load latest model metrics
        self._load_model_metrics()
        
        # 2. Run drift detection
        self._run_drift_detection()
        
        # 3. Update model info
        self._update_model_info()
        
        print("✓ Monitoring cycle completed")
        print("="*60)
    
    def _load_model_metrics(self):
        """Load model metrics from results"""
        try:
            metrics_path = "Results/metrics.txt"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    content = f.read()
                    
                # Parse metrics with better error handling
                if "Accuracy" in content:
                    try:
                        accuracy_part = content.split("Accuracy = ")[1].split(",")[0]
                        accuracy = float(accuracy_part.strip())
                        self.model_accuracy.set(accuracy)
                        print(f"✓ Updated model accuracy: {accuracy:.4f}")
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Could not parse accuracy: {e}")
                
                if "AUC" in content:
                    try:
                        # Better parsing for AUC - handle newlines and extra text
                        auc_part = content.split("AUC = ")[1]
                        # Remove everything after newline or comma
                        auc_part = auc_part.split('\n')[0].split(',')[0].strip()
                        auc = float(auc_part)
                        self.model_auc.set(auc)
                        print(f"✓ Updated model AUC: {auc:.4f}")
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Could not parse AUC: {e}")
                        
        except Exception as e:
            print(f"⚠️ Could not load model metrics: {e}")
    
    def _run_drift_detection(self):
        """Run data drift detection and update metrics"""
        try:
            detector = DataDriftDetector()
            drift_metrics = detector.detect_drift()
            self.update_drift_metrics(drift_metrics)
            print(f"✓ Updated drift metrics - Features drifted: {drift_metrics['n_drifted_features']}")
        except Exception as e:
            print(f"⚠️ Could not run drift detection: {e}")
            # Fallback to default metrics
            drift_metrics = {
                'n_drifted_features': 0,
                'share_drifted_features': 0.0,
                'dataset_drift': False
            }
            self.update_drift_metrics(drift_metrics)
    
    def _update_model_info(self):
        """Update model information"""
        try:
            model_path = "Model/personality_classifier.skops"
            if os.path.exists(model_path):
                # Get file modification time
                mod_time = os.path.getmtime(model_path)
                training_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                self.set_model_info(
                    model_name="RandomForestPersonality",
                    model_version="latest",
                    training_date=training_date
                )
                print(f"✓ Updated model info - Training date: {training_date}")
        except Exception as e:
            print(f"⚠️ Could not update model info: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        print("✓ Monitoring stopped")


def main():
    """Main function to run monitoring"""
    monitoring = MLOpsMonitoring(port=8000)
    
    try:
        # Start monitoring server
        monitoring.start_monitoring()
        
        # Run monitoring cycle
        monitoring.run_monitoring_cycle()
        
        print("\n🔍 Monitoring is now active!")
        print("📊 Prometheus metrics: http://localhost:8000/metrics")
        print("📈 You can now configure Grafana to visualize these metrics")
        print("\nPress Ctrl+C to stop monitoring...")
        
        # Keep monitoring running
        while True:
            time.sleep(60)  # Run monitoring cycle every minute
            monitoring.run_monitoring_cycle()
            
    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")
        monitoring.stop_monitoring()
    except Exception as e:
        print(f"Error in monitoring: {e}")
        monitoring.stop_monitoring()


if __name__ == "__main__":
    main()
