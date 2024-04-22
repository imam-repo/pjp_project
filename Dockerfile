# Base Image
FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Copy Dependency File
COPY requirements.txt requirements.txt

# Install Dependencies 
RUN pip install -r requirements.txt

# Copy your Application Code
COPY . .  

# Expose the API Port
EXPOSE 8000

# Command to Start the API
CMD ["gunicorn", "cluster_api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
