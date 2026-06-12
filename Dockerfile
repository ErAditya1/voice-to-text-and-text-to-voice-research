# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
# ffmpeg is needed for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt httpx

# Copy the rest of the application into the container
COPY . .

# Expose port 9000
EXPOSE 9000

# Set environment variable for models directory inside container
ENV MODELS_DIR="/models"

# Run uvicorn server on port 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
