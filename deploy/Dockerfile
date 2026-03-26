FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create data and logs directories with proper permissions
RUN mkdir -p /app/data /app/logs && \
    chmod 755 /app/data /app/logs

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for database path
ENV DB_PATH=/app/data/tokens.db

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
