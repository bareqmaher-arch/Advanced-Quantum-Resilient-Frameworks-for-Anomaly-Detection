# Q-ZAP: Quantum-Resilient Zero-Trust Anomaly-detection Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org/)
[![TensorFlow Quantum](https://img.shields.io/badge/TFQ-0.7.3-green.svg)](https://www.tensorflow.org/quantum)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

The Q-ZAP framework addresses the dual challenge of securing multi-tenant hybrid cloud environments against both sophisticated contemporary attacks and future quantum threats. This novel architecture synergistically integrates:

- **Post-Quantum Cryptography (PQC)** for cryptographic resilience
- **Hybrid Quantum-Classical Machine Learning (QML)** for enhanced anomaly detection
- **Zero-Trust Architecture (ZTA)** for dynamic policy enforcement

## 🏗️ Architecture

![Q-ZAP Architecture](assets/qzap_architecture.png)

The framework consists of four main layers:

1. **Data Collection Layer**: Aggregates logs and metrics from hybrid infrastructure
2. **PQC-Secured Transport Layer**: Protects data in transit using NIST-standardized algorithms
3. **Hybrid Anomaly Detection Engine**: Core HAE model for real-time threat detection
4. **ZTA Policy Engine**: Dynamic risk-based access control and enforcement

## 🚀 Key Features

- **Quantum-Resistant Security**: Implementation of NIST-standardized PQC algorithms (ML-KEM, ML-DSA, SLH-DSA)
- **Hybrid Autoencoder (HAE)**: Quantum-enhanced anomaly detection with superior performance
- **Multi-Tenant Support**: Federated learning approach preserving tenant privacy
- **Real-time Detection**: Dynamic anomaly scoring with automated policy enforcement
- **Crypto-Agility**: Seamless algorithm replacement and updates

## 📊 Performance Results

| Model | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| Isolation Forest | 0.78 | 0.71 | 0.74 | 0.82 |
| Classical Autoencoder | 0.85 | 0.81 | 0.83 | 0.91 |
| **Q-ZAP HAE** | **0.92** | **0.89** | **0.91** | **0.96** |
| UEBA | 0.75 | 0.79 | 0.77 | 0.85 |

## 🛠️ Installation

### Prerequisites

```bash
Python >= 3.10.4
CUDA-capable GPU (optional, for quantum simulation acceleration)
```

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/bareqmaher-arch/Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection.git
cd Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection
```

2. **Create virtual environment**:
```bash
python -m venv qzap_env
source qzap_env/bin/activate  # On Windows: qzap_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Post-Quantum Cryptography libraries**:
```bash
# Install liboqs (requires custom compilation)
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -GNinja ..
ninja
ninja install

# Install Python bindings
pip install liboqs-python
```

## 📁 Project Structure

```
├── src/
│   ├── qzap/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── hae_model.py          # Hybrid Autoencoder implementation
│   │   │   ├── pqc_utils.py          # Post-Quantum Cryptography utilities
│   │   │   └── zta_engine.py         # Zero-Trust Architecture engine
│   │   ├── models/
│   │   │   ├── quantum_circuit.py    # Quantum circuit definitions
│   │   │   ├── classical_nets.py     # Classical neural networks
│   │   │   └── federated_learning.py # Multi-tenant FL implementation
│   │   ├── security/
│   │   │   ├── pqc_tls.py            # PQC-enabled TLS implementation
│   │   │   ├── crypto_agility.py     # Cryptographic agility framework
│   │   │   └── threat_detection.py   # Anomaly detection algorithms
│   │   └── utils/
│   │       ├── data_processor.py     # Data preprocessing utilities
│   │       ├── metrics.py            # Evaluation metrics
│   │       └── config.py             # Configuration management
├── experiments/
│   ├── datasets/
│   │   └── cic_ids2017_processor.py  # CIC-IDS2017 dataset handler
│   ├── benchmarks/
│   │   ├── anomaly_detection.py      # Detection performance evaluation
│   │   ├── pqc_performance.py        # PQC overhead analysis
│   │   └── end_to_end_test.py        # Complete framework testing
│   └── results/
│       ├── plots/                    # Performance visualization
│       └── logs/                     # Experimental logs
├── deployment/
│   ├── kubernetes/
│   │   ├── qzap-deployment.yaml      # K8s deployment configuration
│   │   └── pqc-configmap.yaml        # PQC configuration
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── terraform/
│       ├── aws-infrastructure.tf     # AWS cloud setup
│       └── variables.tf
├── docs/
│   ├── api/                          # API documentation
│   ├── tutorials/                    # Usage tutorials
│   └── research/                     # Research paper and results
├── tests/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── security/                     # Security tests
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🎯 Quick Start

### 1. Basic Anomaly Detection

```python
from qzap.core.hae_model import HybridAutoencoder
from qzap.utils.data_processor import DataProcessor

# Initialize the Hybrid Autoencoder
hae = HybridAutoencoder(
    input_dim=78,
    latent_dim=4,
    n_qubits=4
)

# Load and preprocess data
processor = DataProcessor()
X_train, X_test = processor.load_cic_ids2017()

# Train the model
hae.fit(X_train, epochs=100, batch_size=32)

# Detect anomalies
anomaly_scores = hae.predict_anomaly_scores(X_test)
```

### 2. PQC-Enabled Communication

```python
from qzap.security.pqc_tls import PQCTLSClient

# Create PQC-enabled client
client = PQCTLSClient(
    cipher_suite='TLS_AES_256_GCM_SHA384:X25519_ML-KEM-768'
)

# Establish secure connection
response = client.get('https://example.com/api/data')
print(f"Status: {response.status_code}")
```

### 3. Complete Framework Deployment

```python
from qzap.core.qzap_framework import QZAPFramework

# Initialize the complete framework
qzap = QZAPFramework(
    config_file='config/production.yaml'
)

# Start monitoring
qzap.start_monitoring()

# Enable real-time threat detection
qzap.enable_anomaly_detection()
```

## 🧪 Running Experiments

### Anomaly Detection Benchmark

```bash
python experiments/benchmarks/anomaly_detection.py \
    --dataset cic_ids2017 \
    --models hae,classical_ae,isolation_forest \
    --output results/anomaly_detection_results.json
```

### PQC Performance Analysis

```bash
python experiments/benchmarks/pqc_performance.py \
    --handshakes 1000 \
    --algorithms classical,hybrid \
    --output results/pqc_performance.json
```

### End-to-End Case Study

```bash
python experiments/benchmarks/end_to_end_test.py \
    --scenario cross_tenant_attack \
    --duration 3600 \
    --output results/end_to_end_results.json
```

## 📊 Monitoring and Visualization

The framework includes a comprehensive monitoring dashboard:

```bash
# Start the monitoring dashboard
python -m qzap.monitoring.dashboard --port 8080

# Access at http://localhost:8080
```

Features:
- Real-time anomaly score visualization
- PQC handshake metrics
- ZTA policy enforcement logs
- Multi-tenant security status

## 🔧 Configuration

### Basic Configuration (`config/default.yaml`)

```yaml
qzap:
  model:
    input_dim: 78
    latent_dim: 4
    n_qubits: 4
    learning_rate: 0.001
    
  pqc:
    key_exchange: "ML-KEM-768"
    signature: "ML-DSA-65"
    fallback_signature: "SLH-DSA-SHAKE-128s"
    
  zta:
    risk_threshold: 0.7
    isolation_timeout: 300
    mfa_required_score: 0.5
    
  federated_learning:
    aggregation_rounds: 10
    local_epochs: 5
    min_clients: 3
```

## 🧬 Quantum Circuit Details

The HAE model uses a variational quantum circuit with the following structure:

```
|0⟩ ─── H ─── RZ(θ₁) ─── ●─────────── ⟨Z⟩
                         │
|0⟩ ─── H ─── RZ(θ₂) ─── X ─── ●───── ⟨Z⟩
                               │
|0⟩ ─── H ─── RZ(θ₃) ─────────  X ─── ⟨Z⟩
```

Where θᵢ values are derived from the classical encoder output.

## 🔒 Security Considerations

- **Quantum-Safe Cryptography**: All communications use NIST-approved PQC algorithms
- **Zero-Trust Principles**: No implicit trust, continuous verification
- **Tenant Isolation**: Strong isolation between multi-tenant environments
- **Crypto-Agility**: Easy algorithm updates and replacements

## 📈 Performance Benchmarks

### Anomaly Detection Performance
- **Precision**: 92% (vs 85% classical)
- **Recall**: 89% (vs 81% classical)
- **F1-Score**: 91% (vs 83% classical)
- **AUC**: 96% (vs 91% classical)

### PQC Overhead
- **Handshake Latency**: +47ms (acceptable for most applications)
- **CPU Usage**: +12% during handshake
- **Throughput**: Minimal impact (<2% reduction)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/

# Type checking
mypy src/
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{qzap2025,
  title={Advanced Quantum-Resilient Frameworks for Anomaly Detection in Multi-Tenant Hybrid Cloud Environments},
  author={[Author Names]},
  journal={International Journal of Computer Applications},
  year={2025},
  volume={X},
  number={Y},
  pages={1--15}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/bareqmaher-arch/Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bareqmaher-arch/Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection/discussions)

## 🗺️ Roadmap

- [ ] **Phase 1**: Core framework implementation ✅
- [ ] **Phase 2**: Hardware quantum device integration
- [ ] **Phase 3**: Advanced QML models (QGNNs, QSVMs)
- [ ] **Phase 4**: Formal verification tools
- [ ] **Phase 5**: Production-ready enterprise features

## 📚 Related Work

- [Post-Quantum Cryptography NIST Standards](https://www.nist.gov/post-quantum-cryptography)
- [TensorFlow Quantum](https://www.tensorflow.org/quantum)
- [Zero Trust Architecture NIST SP 800-207](https://csrc.nist.gov/publications/detail/sp/800-207/final)

---

**Disclaimer**: This is a research prototype. For production use, additional security auditing and testing are recommended.