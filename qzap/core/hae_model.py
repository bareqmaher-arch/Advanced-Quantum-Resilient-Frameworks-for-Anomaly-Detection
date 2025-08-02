"""
Hybrid Quantum-Classical Autoencoder (HAE) for Anomaly Detection
================================================================

This module implements the core HAE model used in the Q-ZAP framework.
The model combines classical neural networks with quantum circuits for
enhanced anomaly detection in multi-tenant cloud environments.

Author: Q-ZAP Research Team
Date: 2025
License: MIT
"""

import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any
import logging
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumCircuitBuilder:
    """Builds parameterized quantum circuits for the HAE model."""
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize the quantum circuit builder.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
        """
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.symbols = sympy.symbols(f'q0:{n_qubits}')
        
    def create_pqc(self) -> cirq.Circuit:
        """
        Create a parameterized quantum circuit (PQC) for the HAE.
        
        Returns:
            cirq.Circuit: The parameterized quantum circuit
        """
        circuit = cirq.Circuit()
        
        # Apply Hadamard gates to create superposition
        for qubit in self.qubits:
            circuit.append(cirq.H(qubit))
        
        # Apply parameterized rotation gates
        for i, (qubit, symbol) in enumerate(zip(self.qubits, self.symbols)):
            circuit.append(cirq.rz(symbol)(qubit))
        
        # Add entangling gates (CNOT chain)
        for i in range(len(self.qubits) - 1):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
        
        # Optional: Add more parameterized layers for expressivity
        for i, (qubit, symbol) in enumerate(zip(self.qubits, self.symbols)):
            circuit.append(cirq.ry(symbol)(qubit))
            
        return circuit
    
    def get_observables(self) -> List[cirq.PauliString]:
        """
        Get the measurement observables for the quantum circuit.
        
        Returns:
            List of Pauli-Z observables for each qubit
        """
        return [cirq.Z(qubit) for qubit in self.qubits]


class ClassicalEncoder(tf.keras.Model):
    """Classical encoder network for the HAE model."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the classical encoder.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space (should match n_qubits)
            hidden_dims: List of hidden layer dimensions
        """
        super(ClassicalEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build the encoder layers
        self.layers_list = []
        
        # Input layer
        self.layers_list.append(tf.keras.layers.Dense(
            hidden_dims[0], 
            activation='relu',
            input_shape=(input_dim,),
            name='encoder_dense_1'
        ))
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims[1:], 2):
            self.layers_list.append(tf.keras.layers.Dense(
                dim, 
                activation='relu',
                name=f'encoder_dense_{i}'
            ))
        
        # Output layer with tanh activation for normalization
        self.layers_list.append(tf.keras.layers.Dense(
            latent_dim, 
            activation='tanh',
            name='encoder_output'
        ))
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass through the encoder."""
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class ClassicalDecoder(tf.keras.Model):
    """Classical decoder network for the HAE model."""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = None):
        """
        Initialize the classical decoder.
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output (reconstruction)
            hidden_dims: List of hidden layer dimensions
        """
        super(ClassicalDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64]
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build the decoder layers
        self.layers_list = []
        
        # Input layer
        self.layers_list.append(tf.keras.layers.Dense(
            hidden_dims[0], 
            activation='relu',
            input_shape=(latent_dim,),
            name='decoder_dense_1'
        ))
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims[1:], 2):
            self.layers_list.append(tf.keras.layers.Dense(
                dim, 
                activation='relu',
                name=f'decoder_dense_{i}'
            ))
        
        # Output layer (no activation for reconstruction)
        self.layers_list.append(tf.keras.layers.Dense(
            output_dim,
            name='decoder_output'
        ))
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass through the decoder."""
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class HybridAutoencoder(tf.keras.Model):
    """
    Hybrid Quantum-Classical Autoencoder for anomaly detection.
    
    This model combines classical neural networks with quantum circuits
    to enhance feature representation and improve anomaly detection
    performance in multi-tenant cloud environments.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        n_qubits: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        anomaly_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize the Hybrid Autoencoder.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            n_qubits: Number of qubits (defaults to latent_dim)
            hidden_dims: Hidden layer dimensions for encoder/decoder
            learning_rate: Learning rate for optimization
            anomaly_threshold: Threshold for anomaly classification
        """
        super(HybridAutoencoder, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits if n_qubits is not None else latent_dim
        self.learning_rate = learning_rate
        self.anomaly_threshold = anomaly_threshold
        
        # Validate dimensions
        if self.n_qubits != self.latent_dim:
            logger.warning(
                f"n_qubits ({self.n_qubits}) != latent_dim ({self.latent_dim}). "
                "Setting n_qubits = latent_dim."
            )
            self.n_qubits = self.latent_dim
        
        # Initialize components
        self.encoder = ClassicalEncoder(input_dim, latent_dim, hidden_dims)
        self.decoder = ClassicalDecoder(latent_dim, input_dim, hidden_dims[::-1] if hidden_dims else None)
        
        # Build quantum circuit
        self.qc_builder = QuantumCircuitBuilder(self.n_qubits)
        self.pqc = self.qc_builder.create_pqc()
        self.observables = self.qc_builder.get_observables()
        
        # Create TFQ PQC layer
        self.pqc_layer = tfq.layers.PQC(
            self.pqc,
            self.observables,
            differentiator=tfq.differentiators.ParameterShift(),
            name='quantum_layer'
        )
        
        # Anomaly detection components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        
        # Compile the model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the HAE model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Reconstructed output tensor
        """
        # Classical encoding
        encoded_classical = self.encoder(inputs, training=training)
        
        # Quantum processing
        # Note: TFQ expects a batch of circuits, so we create dummy circuits
        batch_size = tf.shape(inputs)[0]
        empty_circuit = tfq.convert_to_tensor([cirq.Circuit()] * 1)
        tiled_circuits = tf.tile(empty_circuit, [batch_size])
        
        # Apply quantum layer
        encoded_quantum = self.pqc_layer([tiled_circuits, encoded_classical])
        
        # Classical decoding
        reconstructed = self.decoder(encoded_quantum, training=training)
        
        return reconstructed
    
    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Encode inputs to quantum-enhanced latent representation.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Quantum-enhanced latent representation
        """
        encoded_classical = self.encoder(inputs, training=False)
        
        # Process through quantum layer
        batch_size = tf.shape(inputs)[0]
        empty_circuit = tfq.convert_to_tensor([cirq.Circuit()] * 1)
        tiled_circuits = tf.tile(empty_circuit, [batch_size])
        
        encoded_quantum = self.pqc_layer([tiled_circuits, encoded_classical])
        
        return encoded_quantum
    
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Train the HAE model and fit the anomaly detector.
        
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level
            callbacks: Optional training callbacks
            
        Returns:
            Training history and statistics
        """
        logger.info("Starting HAE training...")
        
        # Normalize input data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the autoencoder
        history = super().fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks or []
        )
        
        # Get quantum-enhanced latent representations
        logger.info("Extracting quantum-enhanced features...")
        latent_representations = self.encode(X_scaled).numpy()
        
        # Train the anomaly detector on the latent space
        logger.info("Training anomaly detector...")
        self.isolation_forest.fit(latent_representations)
        
        # Calculate training statistics
        reconstruction_errors = np.mean(
            (X_scaled - self.predict(X_scaled))**2, axis=1
        )
        
        self.training_stats = {
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors),
            'training_samples': len(X),
            'latent_dim': self.latent_dim,
            'n_qubits': self.n_qubits
        }
        
        self.is_trained = True
        logger.info("HAE training completed successfully!")
        
        return {
            'history': history.history,
            'stats': self.training_stats
        }
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for input data.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Scale input data
        X_scaled = self.scaler.transform(X)
        
        # Get quantum-enhanced latent representations
        latent_representations = self.encode(X_scaled).numpy()
        
        # Calculate isolation forest scores
        isolation_scores = self.isolation_forest.decision_function(latent_representations)
        
        # Calculate reconstruction errors
        reconstructed = self.predict(X_scaled)
        reconstruction_errors = np.mean((X_scaled - reconstructed)**2, axis=1)
        
        # Combine scores (normalize to [0, 1])
        isolation_scores_norm = (isolation_scores - isolation_scores.min()) / (
            isolation_scores.max() - isolation_scores.min() + 1e-8
        )
        reconstruction_scores_norm = (reconstruction_errors - reconstruction_errors.min()) / (
            reconstruction_errors.max() - reconstruction_errors.min() + 1e-8
        )
        
        # Weighted combination
        anomaly_scores = 0.6 * (1 - isolation_scores_norm) + 0.4 * reconstruction_scores_norm
        
        return anomaly_scores
    
    def predict_anomalies(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary anomaly labels.
        
        Args:
            X: Input data
            threshold: Anomaly threshold (uses default if None)
            
        Returns:
            Binary anomaly labels (1 = anomaly, 0 = normal)
        """
        threshold = threshold if threshold is not None else self.anomaly_threshold
        anomaly_scores = self.predict_anomaly_scores(X)
        return (anomaly_scores > threshold).astype(int)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the complete HAE model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(filepath, exist_ok=True)
        
        # Save the Keras model
        self.save_weights(os.path.join(filepath, 'hae_weights'))
        
        # Save the isolation forest
        joblib.dump(
            self.isolation_forest,
            os.path.join(filepath, 'isolation_forest.pkl')
        )
        
        # Save the scaler
        joblib.dump(
            self.scaler,
            os.path.join(filepath, 'scaler.pkl')
        )
        
        # Save model configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'n_qubits': self.n_qubits,
            'learning_rate': self.learning_rate,
            'anomaly_threshold': self.anomaly_threshold,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats
        }
        
        import json
        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved HAE model.
        
        Args:
            filepath: Path to load the model from
        """
        # Load model configuration
        import json
        with open(os.path.join(filepath, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Update model configuration
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.n_qubits = config['n_qubits']
        self.learning_rate = config['learning_rate']
        self.anomaly_threshold = config['anomaly_threshold']
        self.is_trained = config['is_trained']
        self.training_stats = config['training_stats']
        
        # Load the Keras model weights
        self.load_weights(os.path.join(filepath, 'hae_weights'))
        
        # Load the isolation forest
        self.isolation_forest = joblib.load(
            os.path.join(filepath, 'isolation_forest.pkl')
        )
        
        # Load the scaler
        self.scaler = joblib.load(
            os.path.join(filepath, 'scaler.pkl')
        )
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture and training stats.
        
        Returns:
            Dictionary containing model information
        """
        summary = {
            'model_type': 'Hybrid Autoencoder (HAE)',
            'architecture': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'n_qubits': self.n_qubits,
                'quantum_circuit_depth': len(self.pqc.moments),
                'encoder_params': self.encoder.count_params(),
                'decoder_params': self.decoder.count_params(),
                'total_params': self.count_params()
            },
            'training': {
                'is_trained': self.is_trained,
                'learning_rate': self.learning_rate,
                'anomaly_threshold': self.anomaly_threshold
            }
        }
        
        if self.is_trained:
            summary['training'].update(self.training_stats)
        
        return summary


def create_hae_model(
    input_dim: int,
    latent_dim: int = 4,
    hidden_dims: Optional[List[int]] = None,
    learning_rate: float = 0.001,
    **kwargs
) -> HybridAutoencoder:
    """
    Factory function to create a HAE model with default parameters.
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate for optimization
        **kwargs: Additional model parameters
        
    Returns:
        Configured HAE model
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    
    model = HybridAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        **kwargs
    )
    
    logger.info(f"Created HAE model with input_dim={input_dim}, latent_dim={latent_dim}")
    return model


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Add some anomalies
    n_anomalies = 50
    anomaly_indices = np.random.choice(len(X), n_anomalies, replace=False)
    X[anomaly_indices] += np.random.normal(0, 3, (n_anomalies, X.shape[1]))
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    # Create and train model
    hae = create_hae_model(
        input_dim=X.shape[1],
        latent_dim=4,
        learning_rate=0.001
    )
    
    print("Training HAE model...")
    training_results = hae.fit(
        X_train,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Predict anomalies
    print("Predicting anomalies...")
    anomaly_scores = hae.predict_anomaly_scores(X_test)
    anomaly_labels = hae.predict_anomalies(X_test)
    
    print(f"Detected {np.sum(anomaly_labels)} anomalies out of {len(X_test)} samples")
    print(f"Anomaly rate: {np.mean(anomaly_labels):.2%}")
    
    # Print model summary
    print("\nModel Summary:")
    summary = hae.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save model
    hae.save_model("./trained_hae_model")
    print("Model saved successfully!")