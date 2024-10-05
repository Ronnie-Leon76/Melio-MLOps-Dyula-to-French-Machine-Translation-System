# Stage 1: Build
FROM --platform=linux/amd64 python:3.11.8-slim as build
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY ./serve-requirements.txt .

# Install CPU-optimized PyTorch
RUN pip install --no-cache-dir torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
RUN pip install --no-cache-dir -r serve-requirements.txt

# Copy the application files
COPY ./models /app/models
COPY ./main.py /app/main.py

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11.8-slim
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /app /app

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]