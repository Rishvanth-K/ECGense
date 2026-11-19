# ECGense - Flask web app serving the trained CNN-BiLSTM model
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1. Dependencies first: this layer is cached until requirements.txt changes
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. App code, the existing trained model and the sample beats
COPY webapp/ webapp/
COPY ecg_cnn_bilstm.keras .
COPY samples/ samples/

# 3. Build details from Jenkins, shown in the website footer
ARG BUILD_NUMBER=dev
ARG GIT_COMMIT=local
ENV BUILD_NUMBER=${BUILD_NUMBER} GIT_COMMIT=${GIT_COMMIT}

# 4. Run as a normal user, not root
RUN useradd --create-home ecgense
USER ecgense

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "webapp.app:app"]
