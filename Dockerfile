# Use official Python runtime as base image for AMD64 architecture
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirement.txt .

# Install Python dependencies
# Note: Installing without spacy model download to ensure offline compatibility
RUN pip install --no-cache-dir -r requirement.txt

# Copy the entire project structure
COPY process_pdf.py .
COPY main/ ./main/
COPY xgboost_model.joblib .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables for offline operation
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Make sure the script has proper permissions
RUN chmod +x process_pdf.py

# Default command to process PDFs from input directory to output directory
CMD ["python", "process_pdf.py", "--input", "/app/input", "--output", "/app/output"] 