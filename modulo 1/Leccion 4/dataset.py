import numpy as np

def generate_data(num_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic linear dataset for Adaline training."""
    np.random.seed(42)
    x: np.ndarray = np.linspace(-1, 1, num_samples).reshape(-1, 1)
    y: np.ndarray = 2 * x + np.random.normal(scale=0.2, size=x.shape)  # y = 2x + noise
    return x.astype(np.float32), y.astype(np.float32)
