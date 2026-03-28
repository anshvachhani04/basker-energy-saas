FROM python:3.11-slim

LABEL maintainer="BCG Smart Energy Solutions <devops@baskerenergy.ai>"
LABEL description="Basker Energy SaaS — FastAPI Backend"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML core from streamlit app (shared)
COPY ../streamlit_app/ml_core.py ./ml_core.py
COPY . .

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
