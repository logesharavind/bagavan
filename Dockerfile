# Use the official Python image as the base image
FROM python:3.9.7-slim

# Set environment variables for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port your app runs on (if it's not 8501, change it accordingly)
EXPOSE 8501

# Define the command to run your application
CMD ["python", "app.py"]
