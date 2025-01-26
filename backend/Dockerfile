# Builder stage
FROM python:3.11.11 AS builder

# Install Poetry
RUN pip install "poetry==1.4.2"

# Set up Poetry config and install dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config installer.max-workers 4 && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

# Runtime stage
FROM python:3.11.11-slim AS runtime

# Copy built dependencies from the builder stage
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
