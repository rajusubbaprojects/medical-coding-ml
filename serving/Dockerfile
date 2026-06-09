FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./serving/app
COPY src ./src
COPY data/icd10_3char_descriptions.csv ./data/icd10_3char_descriptions.csv

# Download models from GCS at build time
ARG GCS_BUCKET=medical-coding-ml-models
RUN python -c "
from google.cloud import storage
import os
client = storage.Client()
bucket = client.bucket('medical-coding-ml-models')
os.makedirs('models', exist_ok=True)
for name in ['baseline_encounter.joblib', 'baseline_3char.joblib']:
    blob = bucket.blob(f'models/{name}')
    blob.download_to_filename(f'models/{name}')
    print(f'Downloaded {name}')
"

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
