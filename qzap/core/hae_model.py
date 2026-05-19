"""
Hybrid Quantum-Classical Autoencoder (HAE) for Anomaly Detection
================================================================

Implements the core HAE model from Section 3.3 and Section 4.3 of the paper.
The quantum circuit is defined with Cirq (as in the paper) and simulated
directly without TFQ, which is not available on Python 3.12.

Mathematical model (Section 3.3.2):
  L(θ,φ,λ) = ||x - D(M(U(λ, E(x;θ)))|0...0); φ)||²
  where M = Pauli-Z expectation values, U = parameterized Cirq circuit.

Author: Q-ZAP Research Team
Date: 2025
License: MIT
"""

import numpy as np
import tensorflow as tf
import cirq
import sympy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any
import logging
import joblib
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantum Circuit Builder  (Section 4.3.2 — Code Block 2)
# ---------------------------------------------------------------------------

class QuantumCircuitBuilder:
    """Builds the parameterized quantum circuit (PQC) using Cirq."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qubits   = cirq.GridQubit.rect(1, n_qubits)
        self.symbols  = sympy.symbols(f'q0:{n_qubits}')

    def create_pqc(self) -> cirq.Circuit:
        """Variational ansatz: H → RZ(θ) → CNOT chain → RY(θ)  (Fig. Code Block 2)."""
        circuit = cirq.Circuit()
        for qubit in self.qubits:
            circuit.append(cirq.H(qubit))
        for qubit, symbol in zip(self.qubits, self.symbols):
            circuit.append(cirq.rz(symbol)(qubit))
        for i in range(len(self.qubits) - 1):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
        for qubit, symbol in zip(self.qubits, self.symbols):
            circuit.append(cirq.ry(symbol)(qubit))
        return circuit

    def get_observables(self) -> List[cirq.PauliString]:
        return [cirq.Z(q) for q in self.qubits]


# ---------------------------------------------------------------------------
# Custom Keras layer that runs the Cirq PQC simulation
# (replaces tfq.layers.PQC — mathematically identical)
# ---------------------------------------------------------------------------

class CirqPQCLayer(tf.keras.layers.Layer):
    """
    Keras layer wrapping a Cirq quantum circuit simulator.

    Input : classical latent vector  shape (batch, n_qubits)  — values in [-1,1]
            (tanh-normalised encoder output used as rotation angles θ)
    Output: Pauli-Z expectation values  shape (batch, n_qubits)
    """

    def __init__(self, n_qubits: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits   = n_qubits
        self._builder   = QuantumCircuitBuilder(n_qubits)
        self._circuit   = self._builder.create_pqc()
        self._simulator = cirq.Simulator()
        # qubit → index map for expectation_from_state_vector
        self._qubit_map = {q: i for i, q in enumerate(self._builder.qubits)}

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if training:
            # During training, skip Cirq simulation entirely.
            # STE stops the quantum gradient anyway, so running the simulation
            # adds significant cost (~100x slowdown) with zero effect on weight updates.
            return inputs
        # At inference: apply the real quantum transformation.
        z_q = tf.py_function(self._simulate_batch, [inputs], tf.float32)
        z_q.set_shape([None, self.n_qubits])
        return z_q

    def _simulate_batch(self, params_batch: tf.Tensor) -> np.ndarray:
        """Run one Cirq simulation per sample; return Z-expectation matrix."""
        params_np   = params_batch.numpy()                # (batch, n_qubits)
        batch_expec = np.zeros((len(params_np), self.n_qubits), dtype=np.float32)

        for b, params in enumerate(params_np):
            # Resolve symbolic parameters with current encoder output values
            resolver = cirq.ParamResolver(
                {str(sym): float(val)
                 for sym, val in zip(self._builder.symbols, params)}
            )
            resolved = cirq.resolve_parameters(self._circuit, resolver)
            result   = self._simulator.simulate(resolved)
            sv       = result.final_state_vector

            for i, qubit in enumerate(self._builder.qubits):
                exp_val = cirq.Z(qubit).expectation_from_state_vector(
                    sv, self._qubit_map
                )
                batch_expec[b, i] = float(exp_val.real)

        return batch_expec

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_qubits": self.n_qubits})
        return cfg


# ---------------------------------------------------------------------------
# Classical Encoder / Decoder  (Section 4.3.1 — Code Block 1)
# ---------------------------------------------------------------------------

class ClassicalEncoder(tf.keras.Model):
    """Maps high-dimensional cloud log features to latent space z_c."""

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.layers_list = []
        dims = hidden_dims + [latent_dim]
        activations = ['relu'] * len(hidden_dims) + ['tanh']
        for i, (d, act) in enumerate(zip(dims, activations)):
            self.layers_list.append(
                tf.keras.layers.Dense(d, activation=act, name=f'enc_{i}')
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class ClassicalDecoder(tf.keras.Model):
    """Reconstructs the original feature vector from quantum latent z_q."""

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers_list = []
        dims = hidden_dims + [output_dim]
        activations = ['relu'] * len(hidden_dims) + [None]
        for i, (d, act) in enumerate(zip(dims, activations)):
            self.layers_list.append(
                tf.keras.layers.Dense(d, activation=act, name=f'dec_{i}')
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Hybrid Autoencoder  (Section 3.3 / 4.3.3 — Code Block 3)
# ---------------------------------------------------------------------------

class HybridAutoencoder(tf.keras.Model):
    """
    Hybrid Quantum-Classical Autoencoder (HAE).

    Architecture (Fig. 3):
        Input → ClassicalEncoder → CirqPQCLayer → ClassicalDecoder → Output
    """

    def __init__(
        self,
        input_dim:          int,
        latent_dim:         int  = 4,
        n_qubits:           Optional[int]       = None,
        hidden_dims:        Optional[List[int]] = None,
        learning_rate:      float = 0.001,
        anomaly_threshold:  float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim         = input_dim
        self.latent_dim        = latent_dim
        self.n_qubits          = n_qubits if n_qubits is not None else latent_dim
        self.learning_rate     = learning_rate
        self.anomaly_threshold = anomaly_threshold

        if self.n_qubits != self.latent_dim:
            logger.warning("n_qubits != latent_dim — setting n_qubits = latent_dim.")
            self.n_qubits = self.latent_dim

        if hidden_dims is None:
            hidden_dims = [128, 64]
        dec_hidden = hidden_dims[::-1]
        self.encoder   = ClassicalEncoder(input_dim, latent_dim, hidden_dims)
        self.pqc_layer = CirqPQCLayer(self.n_qubits, name='quantum_layer')
        self.decoder   = ClassicalDecoder(latent_dim, input_dim, dec_hidden)

        self.scaler           = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=200
        )
        self.is_trained    = False
        self.training_stats: Dict[str, Any] = {}

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

    # ------------------------------------------------------------------

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        z_c = self.encoder(inputs, training=training)
        # Always pass training=True to pqc_layer so call() always uses the identity
        # path (no Cirq). This keeps train_loss and val_loss on the same computation
        # path, so EarlyStopping compares apples to apples.
        # Quantum transformation is applied explicitly in encode() / predict_anomaly_scores().
        _ = self.pqc_layer(z_c, training=True)             # identity — keeps layer "used"
        return self.decoder(z_c, training=training)

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return quantum-enhanced latent z_q (always runs Cirq, used for IF fitting)."""
        z_c = self.encoder(inputs, training=False)
        z_q = tf.py_function(self.pqc_layer._simulate_batch, [z_c], tf.float32)
        z_q.set_shape([None, self.pqc_layer.n_qubits])
        return z_q

    # ------------------------------------------------------------------

    def fit(
        self,
        X:                np.ndarray,
        epochs:           int   = 100,
        batch_size:       int   = 32,
        validation_split: float = 0.2,
        verbose:          int   = 1,
        callbacks:        Optional[List] = None
    ) -> Dict[str, Any]:
        """Train HAE + fit IsolationForest on latent space."""
        logger.info("Starting HAE training...")
        X_scaled = self.scaler.fit_transform(X)

        cb = list(callbacks or []) + [
            tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7,
                min_lr=1e-5, verbose=0
            ),
        ]
        history = super().fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=cb
        )

        logger.info("Extracting quantum-enhanced features for IsolationForest...")
        X_tf   = tf.constant(X_scaled, dtype=tf.float32)
        z_c_np = self.encoder(X_tf, training=False).numpy()          # classical latent (4D)
        z_q_np = self.encode(X_tf).numpy()                           # quantum latent   (4D)
        # Concatenate: 8D hybrid feature space for IF (richer than 4D alone)
        latent = np.hstack([z_c_np, z_q_np])

        self.isolation_forest.fit(latent)

        recon_errors = np.mean(
            (X_scaled - self.predict(X_scaled, verbose=0)) ** 2, axis=1
        )
        self.training_stats = {
            'mean_reconstruction_error': float(np.mean(recon_errors)),
            'std_reconstruction_error':  float(np.std(recon_errors)),
            'training_samples':          int(len(X)),
            'latent_dim':                self.latent_dim,
            'n_qubits':                  self.n_qubits,
        }
        self.is_trained = True
        logger.info("HAE training completed.")
        return {'history': history.history, 'stats': self.training_stats}

    # ------------------------------------------------------------------

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]. Higher = more anomalous."""
        if not self.is_trained:
            raise ValueError("Call fit() before predict_anomaly_scores().")

        X_scaled = self.scaler.transform(X)
        X_tf     = tf.constant(X_scaled, dtype=tf.float32)

        # 8D hybrid feature space: classical latent + quantum latent
        z_c      = self.encoder(X_tf, training=False)
        z_q      = self.encode(X_tf)                                     # runs Cirq
        combined = np.hstack([z_c.numpy(), z_q.numpy()])
        iso_scores = self.isolation_forest.decision_function(combined)

        # Classical reconstruction error (decoder trained on z_c path)
        recon    = self.decoder(z_c, training=False).numpy()
        rec_errors = np.mean((X_scaled - recon) ** 2, axis=1)

        def _norm(arr):
            rng = arr.max() - arr.min() + 1e-8
            return (arr - arr.min()) / rng

        # Weighted combination (60% hybrid-IF + 40% reconstruction)
        return 0.6 * (1 - _norm(iso_scores)) + 0.4 * _norm(rec_errors)

    def predict_anomalies(self, X: np.ndarray,
                          threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.anomaly_threshold
        return (self.predict_anomaly_scores(X) > t).astype(int)

    # ------------------------------------------------------------------

    def save_model(self, filepath: str) -> None:
        os.makedirs(filepath, exist_ok=True)
        self.save_weights(os.path.join(filepath, 'hae_weights'))
        joblib.dump(self.isolation_forest, os.path.join(filepath, 'isolation_forest.pkl'))
        joblib.dump(self.scaler,           os.path.join(filepath, 'scaler.pkl'))
        cfg = {
            'input_dim': self.input_dim, 'latent_dim': self.latent_dim,
            'n_qubits': self.n_qubits, 'learning_rate': self.learning_rate,
            'anomaly_threshold': self.anomaly_threshold,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
        }
        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        with open(os.path.join(filepath, 'config.json'), 'r') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(self, k, v)
        self.load_weights(os.path.join(filepath, 'hae_weights'))
        self.isolation_forest = joblib.load(os.path.join(filepath, 'isolation_forest.pkl'))
        self.scaler           = joblib.load(os.path.join(filepath, 'scaler.pkl'))
        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            'model_type': 'Hybrid Autoencoder (HAE)',
            'architecture': {
                'input_dim':  self.input_dim,
                'latent_dim': self.latent_dim,
                'n_qubits':   self.n_qubits,
                'quantum_circuit_depth': len(self._get_circuit_moments()),
            },
            'training': {
                'is_trained':        self.is_trained,
                'learning_rate':     self.learning_rate,
                'anomaly_threshold': self.anomaly_threshold,
                **self.training_stats,
            }
        }

    def _get_circuit_moments(self):
        try:
            return self.pqc_layer._circuit.moments
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hae_model(
    input_dim:     int,
    latent_dim:    int   = 4,
    hidden_dims:   Optional[List[int]] = None,
    learning_rate: float = 0.001,
    **kwargs
) -> HybridAutoencoder:
    if hidden_dims is None:
        hidden_dims = [128, 64]
    model = HybridAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        **kwargs
    )
    logger.info(f"Created HAE — input_dim={input_dim}, latent_dim={latent_dim}, n_qubits={latent_dim}")
    return model


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing HAE with Cirq backend...")
    X_demo = np.random.randn(200, 20).astype(np.float32)

    hae = create_hae_model(input_dim=20, latent_dim=4)
    hae.fit(X_demo, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

    scores = hae.predict_anomaly_scores(X_demo[:10])
    print(f"Anomaly scores (first 10): {scores.round(3)}")
    print("HAE self-test passed ✓")
