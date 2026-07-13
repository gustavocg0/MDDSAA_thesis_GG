"""Dense autoencoder for latent-space drift analysis.

Architecture: Input → 64 → LATENT_DIM → 64 → Input
Activation:  ReLU (hidden layers), none (output)
Loss:        MSE
Optimizer:   Adam

The autoencoder is an ANALYTICAL INSTRUMENT only. It is trained on the
temporal protocol's training split to learn the traffic distribution of that
period (which includes both benign and attack traffic present in training).
Its latent space is used for distribution drift measurement (PSI, KS tests).
It is NOT a detection model and does NOT produce anomaly scores.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _DenseAutoencoder(nn.Module):
    """Two-layer symmetric dense autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X_train:    np.ndarray,
    latent_dim: Optional[int] = None,
    epochs:     Optional[int] = None,
    seed:       int = 42,
) -> Tuple["_DenseAutoencoder", Dict]:
    """Train a dense autoencoder on the temporal training split.

    The autoencoder learns to reconstruct the training-period traffic
    distribution in a LATENT_DIM-dimensional latent space. Latent
    representations are later used for drift analysis — NOT anomaly detection.

    Args:
        X_train:    Feature matrix (n_samples, n_features) from the TEMPORAL
                    training split only. Must be preprocessed (imputed, scaled).
        latent_dim: Latent dimension (default: config.LATENT_DIM).
        epochs:     Number of training epochs (default: config.AUTOENCODER_EPOCHS).
        seed:       Random seed for weight initialisation, data shuffling,
                    and training-data subsampling.

    Returns:
        (trained_model, ae_audit)
        trained_model: _DenseAutoencoder in eval mode.
        ae_audit:      Dict with keys n_train_orig, n_train_used,
                       ae_subsampled, final_loss, loss_history.
    """
    from config import (
        AE_MAX_TRAIN_SAMPLES,
        AUTOENCODER_BATCH_SIZE,
        AUTOENCODER_EPOCHS,
        AUTOENCODER_HIDDEN_DIM,
        AUTOENCODER_LR,
        LATENT_DIM,
    )

    # Use `is None` guards — not `or` — so that an explicit 0 is never silently
    # replaced by the config default.
    if latent_dim is None:
        latent_dim = LATENT_DIM
    if epochs is None:
        epochs = AUTOENCODER_EPOCHS
    hidden_dim = AUTOENCODER_HIDDEN_DIM
    batch_size = AUTOENCODER_BATCH_SIZE
    lr         = AUTOENCODER_LR

    # Set all seeds before subsampling so the subsample itself is reproducible.
    # Use torch.manual_seed only — do NOT call np.random.seed here.  The legacy
    # NumPy global RNG was already seeded by runner._set_global_seed; resetting
    # it mid-experiment would break reproducibility for any subsequent legacy
    # numpy random calls in the same run.  Subsampling below uses the new
    # Generator API (np.random.default_rng) which is independent of legacy state.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    n_orig = len(X_train)
    ae_subsampled = False
    if n_orig > AE_MAX_TRAIN_SAMPLES:
        rng_idx = np.random.default_rng(seed)
        idx = rng_idx.choice(n_orig, size=AE_MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[idx]
        ae_subsampled = True
        logger.info(
            "Autoencoder training subsampled: %d → %d rows (seed=%d)",
            n_orig, len(X_train), seed,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Autoencoder device: %s | n_train=%d", device, len(X_train))

    input_dim = X_train.shape[1]
    model = _DenseAutoencoder(input_dim, hidden_dim, latent_dim).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    loss_history: List[float] = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        mean_loss = epoch_loss / len(X_train)
        loss_history.append(mean_loss)
        if epoch % 10 == 0 or epoch == epochs:
            logger.debug("AE epoch %d/%d — loss=%.6f", epoch, epochs, mean_loss)

    model.eval()
    final_loss = loss_history[-1]
    logger.info(
        "Autoencoder trained: input_dim=%d → latent_dim=%d | %d epochs | final_loss=%.6f",
        input_dim, latent_dim, epochs, final_loss,
    )

    ae_audit = {
        "n_train_orig":  n_orig,
        "n_train_used":  len(X_train),
        "ae_subsampled": ae_subsampled,
        "final_loss":    float(final_loss),
        "loss_history":  loss_history,
    }
    return model, ae_audit


def encode(model: _DenseAutoencoder, X: np.ndarray) -> np.ndarray:
    """Project samples into the autoencoder's latent space.

    Args:
        model: Trained _DenseAutoencoder (returned by train_autoencoder).
        X:     Feature matrix (n_samples, n_features). Must use the same
               preprocessing as the training data.

    Returns:
        Latent representation array of shape (n_samples, latent_dim).
    """
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        Z = model.encode(X_tensor).cpu().numpy()

    return Z.astype("float32")
