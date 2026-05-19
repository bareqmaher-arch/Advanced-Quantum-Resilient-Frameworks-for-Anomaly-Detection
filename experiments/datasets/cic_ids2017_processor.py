"""
CIC-IDS2017 Dataset Processor for Q-ZAP Experiments
====================================================
Handles download URL guidance, loading, preprocessing, and
multi-tenant partitioning of the CIC-IDS2017 dataset as described
in Section 5.1 of the paper.

Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html
Expected files: MachineLearningCSV.zip (contains *_ISCX.csv files)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CIC-IDS2017: 78 numerical features used after dropping label + ID columns
FEATURES_TO_DROP = [
    'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
    'Destination Port', 'Protocol', 'Timestamp', 'Label'
]

# Attack families present in CIC-IDS2017
ATTACK_LABELS = {
    'BENIGN': 0,
    'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1, 'DoS Slowhttptest': 1,
    'DDoS': 1,
    'PortScan': 1,
    'FTP-Patator': 1, 'SSH-Patator': 1,
    'Bot': 1,
    'Infiltration': 1,
    'Web Attack – Brute Force': 1, 'Web Attack – XSS': 1, 'Web Attack – Sql Injection': 1,
    'Heartbleed': 1,
}

# Tenant assignment for multi-tenant simulation (paper Section 5.1)
TENANT_ATTACK_ASSIGNMENT = {
    'tenant_a': ['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'FTP-Patator'],
    'tenant_b': ['BENIGN', 'PortScan', 'Bot', 'SSH-Patator'],
    'tenant_c': ['BENIGN', 'DDoS', 'Web Attack – Brute Force', 'Infiltration'],
}


class CICIDS2017Processor:
    """Loads, cleans, and partitions the CIC-IDS2017 dataset."""

    def __init__(self, data_dir: str = "./data/cic_ids2017"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load all CSV files from data_dir into a single DataFrame."""
        csv_files = glob.glob(os.path.join(self.data_dir, "**/*.csv"), recursive=True)
        csv_files += glob.glob(os.path.join(self.data_dir, "*.csv"))
        csv_files = list(set(csv_files))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}.\n"
                "Download the dataset from:\n"
                "  https://www.unb.ca/cic/datasets/ids-2017.html\n"
                "Extract MachineLearningCSV.zip into the data_dir folder."
            )

        logger.info(f"Loading {len(csv_files)} CSV file(s)...")
        dfs = []
        for f in sorted(csv_files):
            try:
                df = pd.read_csv(f, encoding='utf-8', low_memory=False)
                df.columns = df.columns.str.strip()
                dfs.append(df)
                logger.info(f"  {os.path.basename(f)}: {len(df):,} rows")
            except Exception as e:
                logger.warning(f"  Skipping {f}: {e}")

        self._df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows loaded: {len(self._df):,}")
        return self._df

    # ------------------------------------------------------------------
    # Preprocessing (paper Section 5.1)
    # ------------------------------------------------------------------

    def preprocess(
        self,
        df: Optional[pd.DataFrame] = None,
        max_samples: int = 200_000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Clean, scale, and encode the dataset.

        Returns
        -------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)  0=BENIGN, 1=ATTACK
        feature_names : list
        """
        if df is None:
            df = self._df
        if df is None:
            raise RuntimeError("Call load() first.")

        # --- Drop non-feature columns ---
        cols_to_drop = [c for c in FEATURES_TO_DROP if c in df.columns]
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

        labels_raw = df[label_col].astype(str).str.strip()
        feature_df = df.drop(columns=cols_to_drop + [c for c in [label_col] if c not in cols_to_drop],
                             errors='ignore')

        # --- Binary label ---
        y = labels_raw.map(lambda x: 0 if x.upper() == 'BENIGN' else 1).fillna(1).values.astype(int)

        # --- Keep numeric columns only ---
        feature_df = feature_df.select_dtypes(include=[np.number])
        feature_names = list(feature_df.columns)

        # --- Clean infinities and NaN ---
        X = feature_df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Subsample to keep runtime manageable ---
        if len(X) > max_samples:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
            logger.info(f"Subsampled to {max_samples:,} rows")

        # --- Scale ---
        X = self.scaler.fit_transform(X)

        logger.info(
            f"Preprocessed: {X.shape[0]:,} samples × {X.shape[1]} features | "
            f"Benign={np.sum(y==0):,}  Attack={np.sum(y==1):,}"
        )
        return X, y, feature_names

    # ------------------------------------------------------------------
    # Multi-tenant partitioning (paper Section 5.1)
    # ------------------------------------------------------------------

    def create_tenant_splits(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Partition data across three tenants and produce train/test splits
        matching the multi-tenant simulation in Section 5.1.

        Returns a dict:  tenant_name -> {'X_train', 'y_train', 'X_test', 'y_test'}
        """
        if df is None:
            df = self._df
        if df is None:
            raise RuntimeError("Call load() first.")

        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        df = df.copy()
        df[label_col] = df[label_col].astype(str).str.strip()

        tenant_splits: Dict[str, Dict[str, np.ndarray]] = {}

        for tenant, attack_types in TENANT_ATTACK_ASSIGNMENT.items():
            mask = df[label_col].isin(attack_types)
            tenant_df = df[mask].copy()

            if len(tenant_df) == 0:
                logger.warning(f"No data for tenant {tenant}")
                continue

            processor = CICIDS2017Processor.__new__(CICIDS2017Processor)
            processor.data_dir = self.data_dir
            processor._df = tenant_df
            processor.scaler = StandardScaler()

            X, y, _ = processor.preprocess(tenant_df, max_samples=50_000, random_state=random_state)

            # Train only on benign traffic
            benign_mask = y == 0
            X_benign = X[benign_mask]

            X_train, X_test, _, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if y.sum() > 0 else None
            )

            tenant_splits[tenant] = {
                'X_train': X_train[y_test[:len(X_train)] == 0] if len(X_benign) > 0 else X_benign,
                'X_benign_train': X_benign,
                'X_test': X_test,
                'y_test': y_test,
            }
            logger.info(
                f"Tenant '{tenant}': train_benign={len(X_benign):,}  "
                f"test={len(X_test):,}  attacks_in_test={y_test.sum():,}"
            )

        return tenant_splits

    # ------------------------------------------------------------------
    # Convenience: standard train/test split for single-model benchmarks
    # ------------------------------------------------------------------

    def get_train_test_split(
        self,
        test_size: float = 0.3,
        benign_only_train: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X_train, X_test, y_train, y_test).
        If benign_only_train=True, training set contains only normal traffic
        (unsupervised anomaly detection setting — paper Section 3.3.1).
        """
        if self._df is None:
            raise RuntimeError("Call load() first.")

        X, y, _ = self.preprocess()

        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if benign_only_train:
            benign_mask = y_train_all == 0
            X_train = X_train_all[benign_mask]
            y_train = y_train_all[benign_mask]
            logger.info(
                f"Train (benign only): {len(X_train):,} | "
                f"Test: {len(X_test):,} ({y_test.sum():,} attacks)"
            )
        else:
            X_train, y_train = X_train_all, y_train_all

        return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/cic_ids2017"
    processor = CICIDS2017Processor(data_dir=data_dir)
    df = processor.load()
    X_train, X_test, y_train, y_test = processor.get_train_test_split()
    print(f"\nReady for experiments:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print(f"  Attack rate in test: {y_test.mean():.1%}")
