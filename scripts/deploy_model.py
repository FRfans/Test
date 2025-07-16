"""
Model Deployment Script
Deploys trained model to different environments using MLflow
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import json
import argparse
from datetime import datetime
import requests
import time

class ModelDeployer:
    def __init__(self, model_name="personality-classifier"):
        self.model_name = model_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow connection"""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        print(f"🔧 Connected to MLflow: {tracking_uri}")
    
    def get_latest_model_version(self, stage="None"):
        """Get the latest model version from MLflow Model Registry"""
        try:
            latest_versions = self.client.get_latest_versions(
                self.model_name, 
                stages=[stage] if stage != "None" else None
            )
            
            if latest_versions:
                return latest_versions[0]
            else:
                print(f"⚠️ No model versions found for {self.model_name} in stage {stage}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting model version: {e}")
            return None
    
    def transition_model_stage(self, version, new_stage):
        """Transition model to a new stage"""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version.version,
                stage=new_stage
            )
            print(f"✅ Model version {version.version} transitioned to {new_stage}")
            return True
        except Exception as e:
            print(f"❌ Error transitioning model stage: {e}")
            return False
    
    def deploy_to_staging(self):
        """Deploy model to staging environment"""
        print("🚀 Deploying to staging environment...")
        
        # Get the latest model version
        model_version = self.get_latest_model_version("None")
        
        if model_version is None:
            print("❌ No model available for deployment")
            return False
        
        # Transition to Staging
        success = self.transition_model_stage(model_version, "Staging")
        
        if success:
            # Create deployment configuration
            staging_config = {
                "environment": "staging",
                "model_name": self.model_name,
                "model_version": model_version.version,
                "deployment_time": datetime.now().isoformat(),
                "model_uri": f"models:/{self.model_name}/{model_version.version}"
            }
            
            # Save deployment config
            with open("staging_deployment.json", "w") as f:
                json.dump(staging_config, f, indent=2)
            
            print(f"✅ Model deployed to staging")
            print(f"   Model URI: {staging_config['model_uri']}")
            return True
        
        return False
    
    def deploy_to_production(self):
        """Deploy model to production environment"""
        print("🚀 Deploying to production environment...")
        
        # Get the model from staging
        staging_model = self.get_latest_model_version("Staging")
        
        if staging_model is None:
            print("❌ No staging model available for production deployment")
            return False
        
        # Run validation checks
        if not self.validate_model_for_production(staging_model):
            print("❌ Model validation failed")
            return False
        
        # Transition to Production
        success = self.transition_model_stage(staging_model, "Production")
        
        if success:
            # Archive previous production model if exists
            self.archive_previous_production_model()
            
            # Create deployment configuration
            production_config = {
                "environment": "production",
                "model_name": self.model_name,
                "model_version": staging_model.version,
                "deployment_time": datetime.now().isoformat(),
                "model_uri": f"models:/{self.model_name}/Production",
                "previous_version": self.get_previous_production_version()
            }
            
            # Save deployment config
            with open("production_deployment.json", "w") as f:
                json.dump(production_config, f, indent=2)
            
            print(f"✅ Model deployed to production")
            print(f"   Model URI: {production_config['model_uri']}")
            return True
        
        return False
    
    def validate_model_for_production(self, model_version):
        """Validate model before production deployment"""
        print("🔍 Validating model for production...")
        
        try:
            # Load model
            model_uri = f"models:/{self.model_name}/{model_version.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Check if model can make predictions (basic test)
            # This is a simplified validation - you might want more comprehensive tests
            test_input = {
                'Time_spent_Alone': [3.0],
                'Stage_fear': [0],
                'Social_event_attendance': [4.0],
                'Going_outside': [3],
                'Drained_after_socializing': [0],
                'Friends_circle_size': [5.0],
                'Post_frequency': [2.0]
            }
            
            import pandas as pd
            test_df = pd.DataFrame(test_input)
            prediction = model.predict(test_df)
            
            print(f"✅ Model validation passed")
            print(f"   Test prediction: {prediction}")
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def archive_previous_production_model(self):
        """Archive the previous production model"""
        try:
            current_prod = self.get_latest_model_version("Production")
            if current_prod:
                self.transition_model_stage(current_prod, "Archived")
                print(f"📦 Previous production model v{current_prod.version} archived")
        except Exception as e:
            print(f"⚠️ Could not archive previous model: {e}")
    
    def get_previous_production_version(self):
        """Get the previous production model version"""
        try:
            archived_models = self.client.get_latest_versions(
                self.model_name, 
                stages=["Archived"]
            )
            if archived_models:
                return archived_models[0].version
        except:
            pass
        return None
    
    def rollback_production(self):
        """Rollback production to previous version"""
        print("🔄 Rolling back production deployment...")
        
        try:
            # Get current production model
            current_prod = self.get_latest_model_version("Production")
            
            # Get archived model (previous production)
            archived_model = self.get_latest_model_version("Archived")
            
            if not archived_model:
                print("❌ No archived model available for rollback")
                return False
            
            # Archive current production
            if current_prod:
                self.transition_model_stage(current_prod, "Archived")
            
            # Promote archived to production
            success = self.transition_model_stage(archived_model, "Production")
            
            if success:
                rollback_config = {
                    "action": "rollback",
                    "timestamp": datetime.now().isoformat(),
                    "restored_version": archived_model.version,
                    "previous_version": current_prod.version if current_prod else None
                }
                
                with open("rollback_log.json", "w") as f:
                    json.dump(rollback_config, f, indent=2)
                
                print(f"✅ Production rolled back to version {archived_model.version}")
                return True
        
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def get_deployment_status(self):
        """Get current deployment status"""
        print("📊 Current Deployment Status:")
        
        stages = ["Staging", "Production", "Archived"]
        status = {}
        
        for stage in stages:
            model_version = self.get_latest_model_version(stage)
            if model_version:
                status[stage] = {
                    "version": model_version.version,
                    "creation_timestamp": model_version.creation_timestamp,
                    "last_updated_timestamp": model_version.last_updated_timestamp
                }
                print(f"   {stage}: v{model_version.version}")
            else:
                status[stage] = None
                print(f"   {stage}: No model")
        
        return status

def main():
    parser = argparse.ArgumentParser(description="Deploy ML model")
    parser.add_argument("--environment", choices=["staging", "production"], 
                       default="staging", help="Deployment environment")
    parser.add_argument("--action", choices=["deploy", "rollback", "status"], 
                       default="deploy", help="Action to perform")
    parser.add_argument("--model-name", default="personality-classifier", 
                       help="Model name in MLflow registry")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ModelDeployer(args.model_name)
    
    if args.action == "status":
        deployer.get_deployment_status()
    
    elif args.action == "rollback":
        if args.environment == "production":
            deployer.rollback_production()
        else:
            print("❌ Rollback only supported for production environment")
    
    elif args.action == "deploy":
        if args.environment == "staging":
            success = deployer.deploy_to_staging()
        elif args.environment == "production":
            success = deployer.deploy_to_production()
        
        if success:
            print(f"🎉 Deployment to {args.environment} completed successfully!")
        else:
            print(f"❌ Deployment to {args.environment} failed!")
            exit(1)

if __name__ == "__main__":
    main()
