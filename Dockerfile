FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
ENV DEBUG=False

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
