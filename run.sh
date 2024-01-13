#!/bin/sh

# Define the image name
IMAGE_NAME="finance-app-img"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run -p 8501:8501 $IMAGE_NAME