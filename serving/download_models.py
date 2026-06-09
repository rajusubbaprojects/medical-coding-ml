from google.cloud import storage
import os

os.makedirs("models", exist_ok=True)
client = storage.Client()
bucket = client.bucket("medical-coding-ml-models")
for name in ["baseline_encounter.joblib", "baseline_3char.joblib"]:
    blob = bucket.blob(f"models/{name}")
    blob.download_to_filename(f"models/{name}")
    print(f"Downloaded {name}")
