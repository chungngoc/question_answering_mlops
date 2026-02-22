# Use official lightweight Python image.
FROM python:3.13-slim

# Set working directory.
WORKDIR /app

# Copy requirements file.
COPY requirements.txt .

# Clean up apt cache and install necessary system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Default command to run the FastAPI application using Uvicorn.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]