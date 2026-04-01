# Use a slim Python 3.11 base image to keep the final container small.
FROM python:3.11-slim

# Set the working directory where all subsequent commands will run.
WORKDIR /app

# Copy only requirements first so dependency layers can be cached.
COPY server/requirements.txt .

# Install runtime dependencies needed by the API server.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container.
COPY . .

# Expose the default Hugging Face Spaces port.
EXPOSE 7860

# Health check lets orchestrators verify the container is serving traffic.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f https://atul-k-6o-incident-response-env.hf.space/health || exit 1

# Start the FastAPI app with Uvicorn on all interfaces.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
