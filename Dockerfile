# Use official lightweight Python base image compatible with x86_64 & ARM64 (Oracle Cloud / Windows)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=5000

# Install system dependencies required for ASR, audio processing, ffmpeg, and C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Upgrade pip and set up wheels environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Create runtime directories for output chunks and ChromaDB vector store
RUN mkdir -p /app/op /app/chroma_db

# Expose web application port
EXPOSE 5000

# Run the Video Knowledge RAG application
CMD ["python", "main.py", "-H", "0.0.0.0", "-p", "5000"]
