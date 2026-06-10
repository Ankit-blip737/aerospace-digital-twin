FROM python:3.9-slim

WORKDIR /app

COPY ml_api.py .
COPY uav_model.pt .
COPY scaler.pkl .
COPY model_meta.pkl .

RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn scipy pydantic torch --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8000"]
