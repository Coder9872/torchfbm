import torch

# Simple UX helpers
def get_default_device() -> torch.device:
	"""Returns CUDA if available, else CPU."""
	return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_default_dtype() -> torch.dtype:
	"""Default dtype for numerical stability."""
	return torch.float32

def set_seed(seed: int):
	"""Set global torch seed (determinism depends on backend)."""
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

# Re-export main APIs for nicer imports
from .generators import fbm, generate_davies_harte, generate_cholesky
from .processes import fractional_ou_process, geometric_fbm
from .layers import FBMNoisyLinear, FractionalPositionalEmbedding
from .estimators import estimate_hurst
from .rl import FBMActionNoise
from .schedulers import get_hurst_schedule
from .loss import HurstRegularizationLoss, SpectralConsistencyLoss
from .online import CachedFGNGenerator
from .analysis import covariance_matrix, plot_acf, spectral_scaling_factor
from .transforms import fractional_diff
from .augmentations import FractionalNoiseAugmentation
from .sde import NeuralFSDE

__all__ = [
	# Generators
	'fbm', 'generate_davies_harte', 'generate_cholesky',
	# Processes
	'fractional_ou_process', 'geometric_fbm',
	# Neural layers
	'FBMNoisyLinear', 'FractionalPositionalEmbedding',
	# Analysis
	'covariance_matrix', 'plot_acf', 'spectral_scaling_factor',
	# Transforms
	'fractional_diff',
	# Estimators
	'estimate_hurst',
	# Loss functions
	'HurstRegularizationLoss', 'SpectralConsistencyLoss',
	# Augmentations
	'FractionalNoiseAugmentation',
	# Reinforcement Learning
	'FBMActionNoise',
	# Schedulers
	'get_hurst_schedule',
	# Online/Real-time
	'CachedFGNGenerator',
	# Neural SDEs
	'NeuralFSDE',
	# Utilities
	'get_default_device', 'get_default_dtype', 'set_seed'
]
