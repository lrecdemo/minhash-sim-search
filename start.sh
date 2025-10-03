#!/bin/bash

mkdir -p data logs

echo "Building and starting Text Clustering App..."
docker-compose up --build -d

echo "Containers started successfully!"
echo "App is available at: http://localhost:8501"
echo ""
echo "To view logs: docker-compose logs -f text-clustering-app"
echo "To stop: docker-compose down"
echo "To rebuild: docker-compose up --build"