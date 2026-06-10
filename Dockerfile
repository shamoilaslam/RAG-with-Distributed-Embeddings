FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY pyproject.toml .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose ports
EXPOSE 8000 5672 8001

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
