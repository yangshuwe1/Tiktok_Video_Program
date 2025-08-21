FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ffmpeg \
    sox \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    libsndfile1 \
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Additional dependencies for Playwright
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libxss1 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libxss1 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements files for installation
COPY requirements.txt ./requirements.txt

# Install core dependencies first with specific versions
RUN python -m pip install --no-cache-dir --timeout 300 torch>=2.0.0 torchvision>=0.15.0 numpy==1.24.3 pandas Pillow matplotlib>=3.5.0 jupyter ipykernel==6.29.0

# Install remaining dependencies
RUN python -m pip install --no-cache-dir --timeout 300 -r requirements.txt

# Install Playwright browsers (Python Playwright)
RUN python -m playwright install-deps chromium
RUN python -m playwright install chromium

# Create necessary directories
RUN mkdir -p /app/src /app/data /app/models /app/test /app/config /app/logs

EXPOSE 8888