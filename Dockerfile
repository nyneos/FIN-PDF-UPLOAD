FROM python:3.11-slim

# Install system packages required for OCR, PDF processing and OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       wget \
       git \
       poppler-utils \
       tesseract-ocr \
       tesseract-ocr-eng \
       tesseract-ocr-hin \
       libgl1 \
       libsm6 \
       libxext6 \
       libtiff5-dev \
       libjpeg-dev \
       zlib1g-dev \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure tesseract is available in PATH (default /usr/bin)
ENV PATH="/usr/bin:${PATH}"

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
