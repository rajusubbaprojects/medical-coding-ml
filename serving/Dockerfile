FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./serving/app
COPY serving/download_models.py ./download_models.py
COPY src ./src
COPY data/icd10_3char_descriptions.csv ./data/icd10_3char_descriptions.csv

RUN python download_models.py

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
