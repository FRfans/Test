import requests
import json
import os

# Load feature names to ensure consistency
def load_feature_names():
    """Load feature names from saved model"""
    feature_files = [
        "Model/feature_names.json",
        "../Model/feature_names.json",
        "./Model/feature_names.json"
    ]
    
    for file_path in feature_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    
    # Fallback to expected features if file not found
    print("⚠️ Feature names file not found, using default order")
    return [
        'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
        'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
        'Post_frequency'
    ]

# Load expected feature names
expected_features = load_feature_names()
print(f"Expected features: {expected_features}")

# Data yang akan diprediksi - ensure all required features are present
data = [
    {
        "Time_spent_Alone": 4.0,
        "Stage_fear": 0,
        "Social_event_attendance": 4.0,
        "Going_outside": 6.0,
        "Drained_after_socializing": 0,
        "Friends_circle_size": 13.0,
        "Post_frequency": 5.0
    },
    {
        "Time_spent_Alone": 9.0,
        "Stage_fear": 1,
        "Social_event_attendance": 0.0,
        "Going_outside": 0.0,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 0.0,
        "Post_frequency": 3.0
    }
]

# Validate data has all required features
for i, sample in enumerate(data):
    missing_features = set(expected_features) - set(sample.keys())
    if missing_features:
        print(f"❌ Sample {i+1} missing features: {missing_features}")
    else:
        print(f"✅ Sample {i+1} has all required features")

# Endpoint MLflow model
url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

# Format payload
payload = json.dumps({"dataframe_records": data})

print(f"\n🚀 Sending prediction request...")
print(f"Features per sample: {len(expected_features)}")
print(f"Samples: {len(data)}")

# Kirim request POST ke endpoint
try:
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        print("✅ Predictions:", response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")
    print("Make sure MLflow model server is running on http://127.0.0.1:1234")
