import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
import os
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import time
import argparse
from datetime import datetime

# Import skops untuk menyimpan model yang kompatibel dengan Hugging Face Spaces
import skops.io as sio
from skops.io import get_untrusted_types
import skops.io as sio
from skops.io import get_untrusted_types

# Import feature validator
from feature_validator import FeatureValidator

warnings.filterwarnings("ignore")


class PersonalityClassifier:
    def __init__(self, data_path="Data/personality_datasert.csv", retrain=False, old_data_path=None):
        self.data_path = data_path
        self.retrain = retrain
        self.old_data_path = old_data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()


    def load_and_explore_data(self):
        print("Memuat dataset...")

        # Check if data file exists
        if not os.path.exists(self.data_path):
            print(f"❌ Data file not found: {self.data_path}")
            print("🔧 Creating sample dataset for CI/CD...")
            
            # Create sample data for CI/CD with correct feature structure using validator
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            sample_df = FeatureValidator.create_sample_data(n_samples=1000, random_state=42)
            sample_df.to_csv(self.data_path, index=False)
            print(f"✅ Created sample dataset: {sample_df.shape}")
            print(f"✅ Features: {list(sample_df.columns[:-1])}")
            
            # Validate the created sample data
            is_valid, issues = FeatureValidator.validate_dataframe(sample_df, stage="training")
            if not is_valid:
                print(f"⚠️ Sample data validation issues: {issues}")
            else:
                print("✅ Sample data validation passed")

        new_data = pd.read_csv(self.data_path)
        print(f"Data baru: {new_data.shape}")
        
        # Validate the loaded data
        is_valid, issues = FeatureValidator.validate_dataframe(new_data, stage="training")
        if not is_valid:
            print(f"⚠️ Data validation issues: {issues}")
        else:
            print("✅ Data validation passed")

        if self.retrain and self.old_data_path and os.path.exists(self.old_data_path):
            print(f"Gabungkan dengan data lama dari: {self.old_data_path}")
            old_data = pd.read_csv(self.old_data_path)
            self.data = pd.concat([old_data, new_data], ignore_index=True)
            print(f"Total data setelah digabung: {self.data.shape}")
        else:
            self.data = new_data

        print("\nDistribusi target:")
        print(self.data["Personality"].value_counts())


    def preprocess_data(self):
        print("\nPreprocessing data...")
        
        # Standardize the dataframe using the validator
        self.data = FeatureValidator.standardize_dataframe(self.data.dropna())
        
        # Separate features and target
        X = self.data.drop(FeatureValidator.TARGET_FEATURE, axis=1)
        y = self.data[FeatureValidator.TARGET_FEATURE]
        
        # Validate feature structure
        is_valid, issues = FeatureValidator.validate_dataframe(self.data, stage="training")
        if not is_valid:
            print(f"⚠️ Preprocessing validation issues: {issues}")
        else:
            print("✅ Preprocessing validation passed")
        
        print(f"✅ Final features ({len(X.columns)}): {list(X.columns)}")
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    def visualize_data(self):
        print("\nMembuat visualisasi...")
        os.makedirs("Results", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        self.data["Personality"].value_counts().plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Distribusi Personality")
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=axes[0, 1])
            axes[0, 1].set_title("Heatmap Korelasi Fitur")
        features_to_plot = [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
        ]
        for i, feature in enumerate(features_to_plot):
            if feature in self.data.columns:
                row, col = divmod(i + 2, 3)
                if row < 2:
                    sns.boxplot(
                        data=self.data, x="Personality", y=feature, ax=axes[row, col]
                    )
                    axes[row, col].set_title(f"{feature} berdasarkan Personality")
        plt.tight_layout()
        plt.savefig("Results/data_exploration.png", dpi=300, bbox_inches="tight")

    def train_best_model(self):
        print("\nMelatih model Random Forest dengan hyperparameter tuning...")
        best_model = RandomForestClassifier(random_state=42)
        best_params = {
            "model__n_estimators": [100],
            "model__max_depth": [None],
            "model__min_samples_split": [10],
            "model__min_samples_leaf": [1],
        }
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", best_model)])
        grid_search = GridSearchCV(
            pipeline, best_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Best Params: {grid_search.best_params_}")
        model_results = {
            "Random Forest": {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "model": self.best_model,
            }
        }
        return model_results

    def evaluate_model(self, model_results):
        print("\nEvaluasi model...")
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=target_names,
            yticklabels=target_names,
        )
        axes[0].set_title("Confusion Matrix")
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        axes[1].plot([0, 1], [0, 1], linestyle="--")
        axes[1].set_title("ROC Curve")
        importances = self.best_model.named_steps["model"].feature_importances_
        feature_names = self.X_train.columns
        indices = np.argsort(importances)[::-1]
        axes[2].bar(range(len(importances)), importances[indices])
        axes[2].set_title("Feature Importance")
        axes[2].set_xticks(range(len(importances)))
        axes[2].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig("Results/model_evaluation.png", dpi=300, bbox_inches="tight")
        with open("Results/metrics.txt", "w") as outfile:
            outfile.write(f"Accuracy = {round(accuracy, 4)}, AUC = {round(auc_score, 4)}")
            outfile.write(
                f"\nCV Score = {round(model_results['Random Forest']['best_score'], 4)}"
            )
        return accuracy, auc_score

    def save_model(self):
        print("\nMenyimpan model...")
        os.makedirs("Model", exist_ok=True)
        
        # Save model, label encoder, and feature names
        sio.dump(self.best_model, "Model/personality_classifier.skops")
        sio.dump(self.label_encoder, "Model/label_encoder.skops")
        sio.dump(list(self.X_train.columns), "Model/feature_names.skops")
        
        # Also save feature names as JSON for easier access
        import json
        with open("Model/feature_names.json", "w") as f:
            json.dump(list(self.X_train.columns), f, indent=2)
        
        # Save feature schema using validator
        FeatureValidator.save_feature_schema("Model")
        
        print(f"✅ Model & artifacts saved with {len(self.X_train.columns)} features:")
        print(f"   Features: {list(self.X_train.columns)}")

        try:
            # Validate saved model with get_untrusted_types for CI/CD
            untrusted_types = get_untrusted_types(file="Model/personality_classifier.skops")
            loaded_model = sio.load("Model/personality_classifier.skops", trusted=untrusted_types)
            
            # Validate feature count consistency
            loaded_features = sio.load("Model/feature_names.skops", trusted=get_untrusted_types(file="Model/feature_names.skops"))
            expected_features = len(self.X_train.columns)
            
            # Check if the model pipeline expects the correct number of features
            if hasattr(loaded_model, 'named_steps') and 'scaler' in loaded_model.named_steps:
                scaler_features = loaded_model.named_steps['scaler'].n_features_in_
                if scaler_features != expected_features:
                    raise ValueError(f"Scaler expects {scaler_features} features, but training data has {expected_features}")
            
            print(f"✅ Model validation successful - {len(loaded_features)} features")
            print("Model valid dan siap deploy!")
        except Exception as e:
            print(f"❌ Gagal load model: {e}")
            raise

    def run_complete_pipeline(self):
        print("=" * 60)
        print("MLFLOW TRACKING: RandomForest - Personality Classifier")
        print("=" * 60)

        mlflow.set_experiment("Personality Classification")

        with mlflow.start_run(run_name="RandomForest_PersonalityClassifier"):
            # Track training start time
            training_start_time = time.time()
            
            self.load_and_explore_data()
            self.preprocess_data()
            self.visualize_data()

            model_results = self.train_best_model()
            accuracy, auc_score = self.evaluate_model(model_results)

            # Calculate training duration
            training_duration = time.time() - training_start_time

            best_params = model_results["Random Forest"]["best_params"]
            for param, value in best_params.items():
                mlflow.log_param(param, value)

            # Log training metadata
            mlflow.log_param("training_duration", training_duration)
            mlflow.log_param("training_date", datetime.now().isoformat())
            mlflow.log_param("data_size", len(self.data))
            mlflow.log_param("feature_count", len(self.X_train.columns))

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("training_duration_seconds", training_duration)

            # ✅ Tambahkan input example & signature
            input_example = self.X_test.iloc[:5]
            signature = infer_signature(self.X_test, self.best_model.predict(self.X_test))

            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="random_forest_model",  # Tetap gunakan ini agar tidak error
                registered_model_name="RandomForestPersonality",
                input_example=input_example,
                signature=signature
            )

            # Log artifacts
            mlflow.log_artifact("Results/data_exploration.png")
            mlflow.log_artifact("Results/model_evaluation.png")
            mlflow.log_artifact("Results/metrics.txt")

            self.save_model()

            print("\n Tracking model selesai di MLflow!")
            print(" Lihat: http://localhost:5000")
            print(f"Training duration: {training_duration:.2f} seconds")

            return self.best_model, accuracy, auc_score
    
def parse_args():
    parser = argparse.ArgumentParser(description="Training atau retraining Personality Classifier")
    parser.add_argument('--data_path', type=str, default="Data/personality_datasert.csv", help="Path ke data baru")
    parser.add_argument('--old_data_path', type=str, default=None, help="Path ke data lama (jika retrain)")
    parser.add_argument('--retrain', action='store_true', help="Aktifkan retrain dengan data baru")
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        
        # Validate paths exist
        if not os.path.exists(args.data_path):
            print(f"❌ Error: Data file not found: {args.data_path}")
            sys.exit(1)
            
        if args.retrain and args.old_data_path and not os.path.exists(args.old_data_path):
            print(f"❌ Error: Old data file not found: {args.old_data_path}")
            sys.exit(1)
        
        print(f"🚀 Starting training with data: {args.data_path}")
        if args.retrain:
            print(f"🔄 Retraining mode enabled")
            if args.old_data_path:
                print(f"📊 Old data: {args.old_data_path}")
        
        classifier = PersonalityClassifier(
            data_path=args.data_path,
            retrain=args.retrain,
            old_data_path=args.old_data_path
        )
        
        model, acc, auc = classifier.run_complete_pipeline()
        
        print(f"✅ Training completed successfully!")
        print(f"📊 Final Accuracy: {acc:.4f}")
        print(f"📊 Final AUC: {auc:.4f}")
        
        return model, acc, auc
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    model, acc, auc = main()
