"""
Dynamic Data Drift Simulator
Mengubah data sintetik secara berkala untuk simulasi drift
"""
import pandas as pd
import numpy as np
import time
import shutil
from datetime import datetime

def create_dynamic_drift_simulator():
    """Create a script that continuously modifies synthetic data"""
    
    print("🎯 Dynamic Data Drift Simulator")
    print("This will modify synthetic_ctgan_data.csv every 5 minutes")
    print("Press Ctrl+C to stop")
    
    # Backup original
    original_data = pd.read_csv("Data/synthetic_ctgan_data.csv")
    original_data.to_csv("Data/synthetic_ctgan_data_original.csv", index=False)
    
    drift_cycle = 0
    
    try:
        while True:
            drift_cycle += 1
            print(f"\n🔄 Drift Cycle {drift_cycle} - {datetime.now()}")
            
            # Load current data
            current_data = pd.read_csv("Data/synthetic_ctgan_data.csv")
            
            # Apply progressive drift
            drifted_data = current_data.copy()
            
            # Gradually shift numeric columns
            numeric_cols = ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency']
            
            for col in numeric_cols:
                if col in drifted_data.columns:
                    # Add incremental drift
                    drift_amount = drift_cycle * 0.5  # Increase drift over time
                    noise = np.random.normal(drift_amount, 1, len(drifted_data))
                    drifted_data[col] = drifted_data[col] + noise
                    drifted_data[col] = drifted_data[col].clip(lower=0)  # No negative values
            
            # Save modified data
            drifted_data.to_csv("Data/synthetic_ctgan_data.csv", index=False)
            
            print(f"✅ Applied drift cycle {drift_cycle}")
            print(f"📊 Check Grafana for updated metrics")
            print(f"⏱️ Next update in 1 minutes...")

            time.sleep(60)  # Wait 1 minute
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping drift simulator...")
        # Restore original data
        shutil.copy("Data/synthetic_ctgan_data_original.csv", "Data/synthetic_ctgan_data.csv")
        print("✅ Original data restored")

if __name__ == "__main__":
    create_dynamic_drift_simulator()
