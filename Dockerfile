# Use the official Python base image for version 3.12.1
FROM python:3.12.1-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* \

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory in the Docker image
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8502 available to the world outside this container
EXPOSE 8502

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port", "8502"]