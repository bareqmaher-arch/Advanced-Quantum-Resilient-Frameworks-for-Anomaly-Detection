# Q-ZAP Framework Docker Image
# ============================
# Multi-stage build for optimized production image

# Stage 1: Build dependencies and quantum libraries
FROM python:3.11-slim as builder

# Build arguments
ARG BUILD_ENV=production
ARG ENABLE_GPU=false
ARG OQS_VERSION=0.9.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    libssl-dev \
    libffi-dev \
    pkg-config \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Install Python build tools
RUN pip install --upgrade pip setuptools wheel

# Build Open Quantum Safe (OQS) library for Post-Quantum Cryptography
RUN git clone --depth 1 --branch ${OQS_VERSION} https://github.com/open-quantum-safe/liboqs.git && \
    cd liboqs && \
    mkdir build && \
    cd build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DOQS_BUILD_ONLY_LIB=ON \
        -DOQS_MINIMAL_BUILD=OFF \
        .. && \
    ninja && \
    ninja install && \
    ldconfig

# Build Python OQS bindings
RUN git clone --depth 1 https://github.com/open-quantum-safe/liboqs-python.git && \
    cd liboqs-python && \
    pip wheel . -w /build/wheels/

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip wheel -r requirements.txt -w /build/wheels/

# Stage 2: Runtime image
FROM python:3.11-slim as runtime

# Runtime arguments
ARG BUILD_ENV=production
ARG USER_ID=1000
ARG GROUP_ID=1000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    QZAP_ENV=${BUILD_ENV} \
    QZAP_HOME=/app \
    QZAP_DATA_DIR=/app/data \
    QZAP_MODELS_DIR=/app/models \
    QZAP_LOGS_DIR=/app/logs \
    QZAP_CONFIG_DIR=/app/config

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    libffi8 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g ${GROUP_ID} qzap && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash qzap

# Create application directories
RUN mkdir -p /app/{src,data,models,logs,config,keys} && \
    chown -R qzap:qzap /app

# Copy OQS libraries from builder stage
COPY --from=builder /usr/local/lib/liboqs* /usr/local/lib/
COPY --from=builder /usr/local/include/oqs /usr/local/include/oqs
RUN ldconfig

# Switch to application directory
WORKDIR /app

# Copy Python wheels and install
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir --find-links /tmp/wheels \
    $(ls /tmp/wheels/*.whl) && \
    rm -rf /tmp/wheels

# Copy application code
COPY src/ src/
COPY config/ config/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install Q-ZAP package
RUN pip install -e .

# Copy additional configuration files
COPY deployment/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY deployment/docker/healthcheck.sh /usr/local/bin/healthcheck.sh

# Make scripts executable
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create default configuration
RUN cp config/default.yaml config/production.yaml

# Set ownership
RUN chown -R qzap:qzap /app

# Switch to non-root user
USER qzap

# Create necessary directories in user home
RUN mkdir -p ~/.qzap/{config,models,logs,keys}

# Expose ports
EXPOSE 8080 8443 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Volume mounts for persistent data
VOLUME ["/app/data", "/app/models", "/app/logs", "/app/config"]

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["qzap-monitor", "start", "--config", "/app/config/production.yaml"]

# Labels
LABEL maintainer="Q-ZAP Research Team" \
      version="1.0.0" \
      description="Quantum-Resilient Zero-Trust Anomaly-detection Platform" \
      org.opencontainers.image.title="Q-ZAP Framework" \
      org.opencontainers.image.description="Hybrid quantum-classical security framework" \
      org.opencontainers.image.vendor="Q-ZAP Research Team" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/bareqmaher-arch/Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection" \
      org.opencontainers.image.licenses="MIT"