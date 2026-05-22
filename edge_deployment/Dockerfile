# Use the official Python 3.10 slim image (Simulates a clean, 64-bit Linux Edge OS)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system libraries needed for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install AI dependencies directly
RUN pip install --no-cache-dir torch torchvision opencv-python-headless numpy pillow

# Copy our entire pipeline into the Edge container
COPY . /app/

# Ensure output directory exists
RUN mkdir -p /app/Fire_Alerts

# The command that runs automatically when the container starts
CMD ["python", "step3_edge_tracker.py"]
