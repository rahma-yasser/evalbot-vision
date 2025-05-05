# Use a base image with Python and necessary dependencies
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy bootstrap.py and requirements.txt
COPY bootstrap.py requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run expects
ENV PORT=8080
EXPOSE $PORT

# Create directory for the model
RUN mkdir -p /models

# Run the bootstrap script
CMD ["python", "bootstrap.py"]