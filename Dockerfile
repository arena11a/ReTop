FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tokenizers fastapi uvicorn

# Copy project
COPY hmn/ hmn/
COPY retop_tokenizer.json .
COPY hmn_v33.pt .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API
CMD ["python", "-m", "hmn.api"]
