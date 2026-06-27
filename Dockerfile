FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for standard pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code directly into WORKDIR (/app)
COPY backend/ .

# Create the user UID 1000 for Hugging Face Spaces permissions
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH=/app

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
