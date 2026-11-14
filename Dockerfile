
FROM python:3.12-slim

WORKDIR /app

# Install application dependencies

COPY requirements-docker.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-docker.txt


# Copy application code and saved model artifacts
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose the FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
