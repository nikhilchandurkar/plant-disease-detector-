# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but useful for image processing libs)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model
COPY app.py .
COPY plant_disease_model.pth .

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
