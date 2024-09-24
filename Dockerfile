# Use an official Python runtime as a parent image, specifying the platform
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Update package lists and install dependencies (OpenMP and gcc for LightGBM)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    gcc \
    gnupg2 \
    ca-certificates \
    curl \
    apt-transport-https && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask app
ENV FLASK_APP=app.py

# Run the application
CMD ["python", "app.py"]
