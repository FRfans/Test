import time
import os
import pandas as pd
from datetime import datetime
from synthetic_generator import PersonalitySyntheticDataGenerator  # pastikan sudah ada

OUTPUT_DIR = "Data/versions"
INTERVAL = 60  # detik

def continuously_generate():
    print("📢 Mulai proses penambahan data sintetik terus-menerus...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generator = PersonalitySyntheticDataGenerator()
    count = 0

    while True:
        try:
            synthetic_data = generator.generate_ctgan_synthetic_data(n_samples=100)
            if synthetic_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"synthetic_{timestamp}.csv"
                filepath = os.path.join(OUTPUT_DIR, filename)
                synthetic_data.to_csv(filepath, index=False)
                count += 1
                print(f"✅ [{count}] Data batch disimpan: {filepath}")
        except Exception as e:
            print(f"❌ Error saat generate: {e}")
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    continuously_generate()
