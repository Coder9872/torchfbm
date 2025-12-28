import torch


# Simple UX helpers
def get_default_device() -> torch.device:
    """Returns CUDA if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_default_dtype() -> torch.dtype:
    """Default dtype for numerical stability."""
    return torch.float32


def set_seed(seed: int):
    """Set global torch seed (determinism depends on backend)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(tensor: torch.Tensor):
    """Helper to convert tensor to numpy with proper error handling."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            raise ImportError(
                "NumPy conversion requested but NumPy is not properly installed or has compatibility issues. "
                "Please install/upgrade numpy: pip install -U numpy"
            ) from e
        raise


# Re-export main APIs for nicer imports
from .generators import fbm, generate_davies_harte, generate_cholesky
from .processes import (
    fractional_ou_process,
    geometric_fbm,
    reflected_fbm,
    fractional_brownian_bridge,
    multifractal_random_walk,
)
from .dfa import dfa
from .layers import (
    FBMNoisyLinear,
    FractionalPositionalEmbedding,
    FractionalKernel,
    fractional_init_,
)
from .estimators import estimate_hurst
from .rl import FBMActionNoise
from .schedulers import get_hurst_schedule
from .loss import HurstRegularizationLoss, SpectralConsistencyLoss
from .online import CachedFGNGenerator
from .analysis import covariance_matrix, plot_acf, spectral_scaling_factor
from .transforms import fractional_diff, fractional_integrate
from .augmentations import FractionalNoiseAugmentation
from .sde import NeuralFSDE

__all__ = [
    # Generators
    "fbm",
    "generate_davies_harte",
    "generate_cholesky",
    # Processes
    "fractional_ou_process",
    "geometric_fbm",
    "reflected_fbm",
    "fractional_brownian_bridge",
    "multifractal_random_walk",
    # Neural layers
    "FBMNoisyLinear",
    "FractionalPositionalEmbedding",
    "FractionalKernel",
    "fractional_init_",
    # Analysis
    "covariance_matrix",
    "plot_acf",
    "spectral_scaling_factor",
    "dfa",
    # Transforms
    "fractional_diff",
    "fractional_integrate",
    # Estimators
    "estimate_hurst",
    # Loss functions
    "HurstRegularizationLoss",
    "SpectralConsistencyLoss",
    # Augmentations
    "FractionalNoiseAugmentation",
    # Reinforcement Learning
    "FBMActionNoise",
    # Schedulers
    "get_hurst_schedule",
    # Online/Real-time
    "CachedFGNGenerator",
    # Neural SDEs
    "NeuralFSDE",
    # Utilities
    "get_default_device",
    "get_default_dtype",
    "set_seed",
]
