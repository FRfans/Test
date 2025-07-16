"""
Enhanced Training Script with MLflow Integration
Integrates model training with MLflow for experiment tracking and model versioning
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
import json
import hashlib

warnings.filterwarnings("ignore")

class MLflowPersonalityTrainer:
    def __init__(self, experiment_name="personality-classification"):
        self.experiment_name = experiment_name
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
        # Set up MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow experiment and tracking"""
        print("🔧 Setting up MLflow...")
        
        # Get MLflow tracking URI from environment or use default
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✅ Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"✅ Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            print(f"⚠️ MLflow setup warning: {e}")
            print("📝 Using local MLflow tracking")
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

    def load_data(self, data_path="Data/synthetic_ctgan_data.csv"):
        """Load and prepare data"""
        print(f"📂 Loading data from {data_path}...")
        
        if not os.path.exists(data_path):
            print(f"⚠️ Synthetic data not found, trying original data...")
            data_path = "Data/personality_datasert.csv"
            
        data = pd.read_csv(data_path)
        print(f"✅ Data loaded: {data.shape}")
        
        # Log data information
        data_info = {
            "data_path": data_path,
            "data_shape": data.shape,
            "columns": list(data.columns),
            "data_hash": self.calculate_data_hash(data)
        }
        
        return data, data_info

    def calculate_data_hash(self, data):
        """Calculate hash of data for tracking changes"""
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()[:10]

    def preprocess_data(self, data):
        """Preprocess data for training"""
        print("🔄 Preprocessing data...")
        
        # Separate features and target
        X = data.drop('Personality', axis=1)
        y = data['Personality']
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"✅ Features: {len(X.columns)}")
        print(f"✅ Target classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded

    def train_model(self, X, y, hyperparameter_tuning=True):
        """Train model with optional hyperparameter tuning"""
        print("🚀 Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if hyperparameter_tuning:
            print("🔍 Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"✅ Best parameters: {best_params}")
            
        else:
            print("🏃 Training with default parameters...")
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=42
            )
            self.model.fit(X_train, y_train)
            best_params = {}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'classification_report': report,
            'best_params': best_params,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def log_to_mlflow(self, data_info, training_results, additional_metrics=None):
        """Log training results to MLflow"""
        print("📝 Logging to MLflow...")
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(training_results['best_params'])
            mlflow.log_param("data_hash", data_info['data_hash'])
            mlflow.log_param("data_shape", str(data_info['data_shape']))
            mlflow.log_param("feature_count", len(self.feature_names))
            
            # Log metrics
            mlflow.log_metric("accuracy", training_results['accuracy'])
            
            if additional_metrics:
                for key, value in additional_metrics.items():
                    mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=training_results['model'],
                artifact_path="model",
                registered_model_name="personality-classifier"
            )
            
            # Log feature names
            with open("feature_names.json", "w") as f:
                json.dump(self.feature_names, f)
            mlflow.log_artifact("feature_names.json")
            
            # Log label encoder
            joblib.dump(self.label_encoder, "label_encoder.joblib")
            mlflow.log_artifact("label_encoder.joblib")
            
            # Create and log confusion matrix
            self.create_confusion_matrix(
                training_results['y_test'], 
                training_results['y_pred']
            )
            mlflow.log_artifact("confusion_matrix.png")
            
            # Log classification report
            with open("classification_report.txt", "w") as f:
                f.write(training_results['classification_report'])
            mlflow.log_artifact("classification_report.txt")
            
            # Log model summary
            model_summary = {
                "model_type": type(training_results['model']).__name__,
                "feature_importances": dict(zip(
                    self.feature_names,
                    training_results['model'].feature_importances_.tolist()
                )),
                "training_timestamp": datetime.now().isoformat(),
                "accuracy": float(training_results['accuracy'])
            }
            
            with open("model_summary.json", "w") as f:
                json.dump(model_summary, f, indent=2)
            mlflow.log_artifact("model_summary.json")
            
            print(f"✅ Logged to MLflow run: {run.info.run_id}")
            return run.info.run_id

    def create_confusion_matrix(self, y_true, y_pred):
        """Create and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_model_artifacts(self):
        """Save model artifacts locally"""
        print("💾 Saving model artifacts...")
        
        os.makedirs("Model", exist_ok=True)
        
        # Save model
        joblib.dump(self.model, "Model/personality_classifier.joblib")
        
        # Save label encoder
        joblib.dump(self.label_encoder, "Model/label_encoder.joblib")
        
        # Save feature names
        with open("Model/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        
        print("✅ Model artifacts saved to Model/ directory")

    def compare_with_previous_models(self):
        """Compare current model with previous models in MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.accuracy DESC"],
                    max_results=5
                )
                
                if len(runs) > 1:
                    print("\n📊 Model Comparison (Top 5 runs by accuracy):")
                    for i, run in enumerate(runs):
                        accuracy = run.data.metrics.get('accuracy', 'N/A')
                        timestamp = datetime.fromtimestamp(run.info.start_time / 1000)
                        print(f"   {i+1}. Run {run.info.run_id[:8]}: {accuracy:.4f} ({timestamp.strftime('%Y-%m-%d %H:%M')})")
                
        except Exception as e:
            print(f"⚠️ Could not compare with previous models: {e}")

def main():
    """Main training pipeline"""
    print("🎯 Starting MLflow-integrated training pipeline...")
    
    # Initialize trainer
    trainer = MLflowPersonalityTrainer()
    
    # Load data
    data, data_info = trainer.load_data()
    
    # Preprocess data
    X, y = trainer.preprocess_data(data)
    
    # Train model
    training_results = trainer.train_model(X, y, hyperparameter_tuning=True)
    
    # Log to MLflow
    run_id = trainer.log_to_mlflow(data_info, training_results)
    
    # Save artifacts locally
    trainer.save_model_artifacts()
    
    # Compare with previous models
    trainer.compare_with_previous_models()
    
    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"📝 MLflow run ID: {run_id}")
    print(f"📊 Final accuracy: {training_results['accuracy']:.4f}")
    print(f"💾 Model artifacts saved locally")

if __name__ == "__main__":
    main()
