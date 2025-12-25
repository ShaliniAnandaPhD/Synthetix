# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sse-starlette \
    google-cloud-texttospeech \
    confluent-kafka \
    python-multipart

# Copy the services directory
COPY services/ ./services/

# Expose the port (Cloud Run sets $PORT environment variable)
ENV PORT=8080

# Run the application
CMD exec uvicorn services.dashboard_api:app --host 0.0.0.0 --port ${PORT}
