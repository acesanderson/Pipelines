FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
   chafa \
   git \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml and install Python dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy the script
COPY dev_script.py .

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs

# Default command
CMD ["python", "dev_script.py", "--prompt", "a cat", "--output", "/app/outputs/generated.png"]
