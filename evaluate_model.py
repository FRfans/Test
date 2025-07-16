"""
Model Evaluation Script
Evaluates trained model performance and generates reports
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

class ModelEvaluator:
    def __init__(self, model_path="Model/personality_classifier.joblib",
                 label_encoder_path="Model/label_encoder.joblib",
                 test_data_path="Data/synthetic_ctgan_data.csv"):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.test_data_path = test_data_path
        self.model = None
        self.label_encoder = None
        
    def load_model_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        print("📂 Loading model artifacts...")
        
        # Load model
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load label encoder
        if os.path.exists(self.label_encoder_path):
            self.label_encoder = joblib.load(self.label_encoder_path)
            print(f"✅ Label encoder loaded from {self.label_encoder_path}")
        else:
            raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")
    
    def load_test_data(self):
        """Load test data"""
        print(f"📂 Loading test data from {self.test_data_path}...")
        
        if os.path.exists(self.test_data_path):
            data = pd.read_csv(self.test_data_path)
            print(f"✅ Test data loaded: {data.shape}")
            return data
        else:
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
    
    def preprocess_data(self, data):
        """Preprocess data for evaluation"""
        print("🔄 Preprocessing test data...")
        
        # Separate features and target
        X = data.drop('Personality', axis=1)
        y = data['Personality']
        
        # Handle categorical features (same as in training)
        from sklearn.preprocessing import LabelEncoder
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Encode target variable
        y_encoded = self.label_encoder.transform(y)
        
        print(f"✅ Data preprocessed: {X.shape}")
        return X, y_encoded, y
    
    def evaluate_model_performance(self, X, y_encoded, y_original):
        """Evaluate model performance with comprehensive metrics"""
        print("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='weighted')
        recall = recall_score(y_encoded, y_pred, average='weighted')
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        
        # For binary classification, calculate AUC
        if len(self.label_encoder.classes_) == 2:
            auc = roc_auc_score(y_encoded, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5, scoring='accuracy')
        
        # Feature importance
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            },
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'classification_report': classification_report(y_encoded, y_pred, 
                                                         target_names=self.label_encoder.classes_,
                                                         output_dict=True)
        }
        
        print(f"✅ Model evaluation completed")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return evaluation_results, y_pred, y_pred_proba
    
    def create_visualizations(self, y_true, y_pred, y_pred_proba, save_path="Results"):
        """Create evaluation visualizations"""
        print("📈 Creating evaluation visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        feature_importance = dict(zip(
            [f'Feature_{i}' for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        ))
        
        plt.figure(figsize=(10, 6))
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, np.array(importance)[sorted_idx])
        plt.yticks(pos, np.array(features)[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curve (for binary classification)
        if len(self.label_encoder.classes_) == 2:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Class Distribution
        plt.figure(figsize=(8, 6))
        true_dist = pd.Series(y_true).value_counts()
        pred_dist = pd.Series(y_pred).value_counts()
        
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.35
        
        true_counts = [true_dist.get(i, 0) for i in range(len(self.label_encoder.classes_))]
        pred_counts = [pred_dist.get(i, 0) for i in range(len(self.label_encoder.classes_))]
        
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('True vs Predicted Class Distribution')
        plt.xticks(x, self.label_encoder.classes_)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {save_path}/")
    
    def save_evaluation_report(self, evaluation_results, save_path="Results"):
        """Save evaluation report"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save JSON report
        with open(f'{save_path}/evaluation_report.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save text report
        with open(f'{save_path}/evaluation_summary.txt', 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Timestamp: {evaluation_results['timestamp']}\n")
            f.write(f"Test Samples: {evaluation_results['test_samples']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            for metric, value in evaluation_results['metrics'].items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            
            f.write("\nFEATURE IMPORTANCE\n")
            f.write("-" * 20 + "\n")
            sorted_features = sorted(evaluation_results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                f.write(f"{feature}: {importance:.4f}\n")
        
        print(f"✅ Evaluation report saved to {save_path}/")
    
    def compare_with_mlflow_models(self, current_metrics):
        """Compare current model with MLflow tracked models"""
        try:
            print("🔍 Comparing with MLflow models...")
            
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment = mlflow.get_experiment_by_name("personality-classification")
            if experiment:
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.accuracy DESC"],
                    max_results=10
                )
                
                comparison_data = []
                for run in runs:
                    run_metrics = run.data.metrics
                    comparison_data.append({
                        'run_id': run.info.run_id,
                        'accuracy': run_metrics.get('accuracy', 0),
                        'timestamp': datetime.fromtimestamp(run.info.start_time / 1000).isoformat()
                    })
                
                # Add current evaluation
                comparison_data.insert(0, {
                    'run_id': 'current_evaluation',
                    'accuracy': current_metrics['accuracy'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save comparison
                with open('Results/model_comparison.json', 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                
                print("✅ Model comparison completed")
                return comparison_data
            
        except Exception as e:
            print(f"⚠️ Could not compare with MLflow models: {e}")
            return None

def main():
    """Main evaluation pipeline"""
    print("🎯 Starting model evaluation pipeline...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model artifacts
        evaluator.load_model_artifacts()
        
        # Load test data
        test_data = evaluator.load_test_data()
        
        # Preprocess data
        X, y_encoded, y_original = evaluator.preprocess_data(test_data)
        
        # Evaluate model
        evaluation_results, y_pred, y_pred_proba = evaluator.evaluate_model_performance(
            X, y_encoded, y_original
        )
        
        # Create visualizations
        evaluator.create_visualizations(y_encoded, y_pred, y_pred_proba)
        
        # Save evaluation report
        evaluator.save_evaluation_report(evaluation_results)
        
        # Compare with MLflow models
        evaluator.compare_with_mlflow_models(evaluation_results['metrics'])
        
        print("\n🎉 Model evaluation completed successfully!")
        print(f"📊 Final Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        print("📁 Results saved to Results/ directory")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
