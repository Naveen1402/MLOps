# Base image
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY tests/ ./tests

CMD ["python", "app/main.py"]
