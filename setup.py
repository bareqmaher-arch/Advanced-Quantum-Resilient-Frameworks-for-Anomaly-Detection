"""
Setup script for Q-ZAP Framework
================================

This script handles the installation and configuration of the 
Quantum-Resilient Zero-Trust Anomaly-detection Platform (Q-ZAP).

Usage:
    pip install -e .                    # Development installation
    python setup.py install             # Standard installation
    python setup.py sdist bdist_wheel   # Build distribution packages
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

# Package metadata
PACKAGE_NAME = "qzap"
VERSION = "1.0.0"
DESCRIPTION = "Quantum-Resilient Zero-Trust Anomaly-detection Platform"
LONG_DESCRIPTION = """
Q-ZAP is a novel security framework that integrates Post-Quantum Cryptography (PQC), 
Hybrid Quantum-Classical Machine Learning (QML), and Zero-Trust Architecture (ZTA) 
to provide comprehensive security for multi-tenant hybrid cloud environments.

Key Features:
- Quantum-resistant cryptography using NIST-standardized algorithms
- Hybrid autoencoder for enhanced anomaly detection
- Dynamic policy enforcement with Zero-Trust principles
- Multi-tenant federated learning support
- Real-time threat detection and response
"""

AUTHOR = "Q-ZAP Research Team"
AUTHOR_EMAIL = "qzap-research@example.com"
URL = "https://github.com/bareqmaher-arch/Advanced-Quantum-Resilient-Frameworks-for-Anomaly-Detection"
LICENSE = "MIT"

# Python version requirement
PYTHON_REQUIRES = ">=3.10"

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    """Read requirements from file."""
    requirements = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
    return requirements

# Core requirements (subset of requirements.txt for basic functionality)
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.1.0",
    "tensorflow>=2.15.0",
    "tensorflow-quantum>=0.7.3",
    "cirq>=1.2.0",
    "cryptography>=41.0.0",
    "requests>=2.28.0",
    "pyyaml>=6.0",
    "loguru>=0.7.0",
    "click>=8.1.0",
    "pydantic>=2.0.0"
]

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    "quantum": [
        "pennylane>=0.30.0",
        "qiskit>=0.45.0"
    ],
    "crypto": [
        "pycryptodome>=3.18.0",
        # "liboqs-python>=0.9.0"  # Requires custom build
    ],
    "cloud": [
        "boto3>=1.28.0",
        "google-cloud>=0.34.0",
        "azure-identity>=1.13.0"
    ],
    "web": [
        "flask>=2.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0"
    ],
    "monitoring": [
        "prometheus-client>=0.17.0",
        "elasticsearch>=8.8.0",
        "grafana-api>=1.0.3"
    ],
    "kubernetes": [
        "kubernetes>=27.2.0",
        "docker>=6.1.0"
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "jupyter>=1.0.0"
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "mkdocs>=1.5.0"
    ],
    "all": [
        # All optional dependencies
        "pennylane>=0.30.0",
        "qiskit>=0.45.0",
        "pycryptodome>=3.18.0",
        "boto3>=1.28.0",
        "google-cloud>=0.34.0",
        "azure-identity>=1.13.0",
        "flask>=2.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "prometheus-client>=0.17.0",
        "elasticsearch>=8.8.0",
        "kubernetes>=27.2.0",
        "docker>=6.1.0",
        "pytest>=7.4.0",
        "jupyter>=1.0.0"
    ]
}

# Console scripts for command-line tools
CONSOLE_SCRIPTS = [
    "qzap=qzap.cli.main:cli",
    "qzap-train=qzap.cli.train:main",
    "qzap-deploy=qzap.cli.deploy:main",
    "qzap-monitor=qzap.cli.monitor:main",
    "qzap-benchmark=qzap.cli.benchmark:main"
]

# Package data
PACKAGE_DATA = {
    "qzap": [
        "config/*.yaml",
        "config/*.json",
        "data/*.csv",
        "models/*.h5",
        "templates/*.html",
        "static/css/*.css",
        "static/js/*.js"
    ]
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Security :: Cryptography",
    "Topic :: System :: Networking :: Monitoring",
    "Topic :: System :: Systems Administration"
]

# Keywords for discovery
KEYWORDS = [
    "quantum-computing",
    "post-quantum-cryptography", 
    "zero-trust",
    "anomaly-detection",
    "machine-learning",
    "cybersecurity",
    "cloud-security",
    "hybrid-quantum",
    "multi-tenant",
    "federated-learning"
]


class CustomInstallCommand(install):
    """Custom install command with post-installation setup."""
    
    def run(self):
        # Run standard installation
        install.run(self)
        
        # Post-installation setup
        self.post_install()
    
    def post_install(self):
        """Perform post-installation setup."""
        print("Setting up Q-ZAP framework...")
        
        # Create necessary directories
        directories = [
            os.path.expanduser("~/.qzap"),
            os.path.expanduser("~/.qzap/config"),
            os.path.expanduser("~/.qzap/models"),
            os.path.expanduser("~/.qzap/logs"),
            os.path.expanduser("~/.qzap/keys")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Copy default configuration
        try:
            import qzap
            config_source = os.path.join(os.path.dirname(qzap.__file__), "config", "default.yaml")
            config_dest = os.path.expanduser("~/.qzap/config/default.yaml")
            
            if os.path.exists(config_source) and not os.path.exists(config_dest):
                import shutil
                shutil.copy2(config_source, config_dest)
                print(f"Copied default configuration to: {config_dest}")
        except Exception as e:
            print(f"Warning: Could not copy default configuration: {e}")
        
        print("Q-ZAP framework setup completed!")
        print("\nNext steps:")
        print("1. Configure your environment: qzap config init")
        print("2. Generate PQC keys: qzap crypto generate-keys")
        print("3. Train the HAE model: qzap-train --dataset your_data.csv")
        print("4. Start monitoring: qzap-monitor start")


class CustomDevelopCommand(develop):
    """Custom develop command for development installation."""
    
    def run(self):
        # Run standard development installation
        develop.run(self)
        
        # Development-specific setup
        self.dev_setup()
    
    def dev_setup(self):
        """Perform development-specific setup."""
        print("Setting up Q-ZAP for development...")
        
        # Install pre-commit hooks if available
        try:
            import subprocess
            subprocess.run(["pre-commit", "install"], check=True)
            print("Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not install pre-commit hooks")
        
        print("Development setup completed!")


def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Q-ZAP requires Python 3.10 or higher. "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )


def main():
    """Main setup function."""
    # Check Python version
    check_python_version()
    
    # Setup configuration
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        
        # Package configuration
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        package_data=PACKAGE_DATA,
        include_package_data=True,
        
        # Dependencies
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        
        # Console scripts
        entry_points={
            "console_scripts": CONSOLE_SCRIPTS
        },
        
        # Metadata
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        
        # Custom commands
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand
        },
        
        # Additional options
        zip_safe=False,
        test_suite="tests",
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}/docs",
            "Research Paper": f"{URL}/blob/main/docs/research/paper.pdf"
        }
    )


if __name__ == "__main__":
    main()