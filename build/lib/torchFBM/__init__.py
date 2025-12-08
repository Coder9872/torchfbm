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

__all__ = [
	'fbm', 'generate_davies_harte', 'generate_cholesky',
	'fractional_ou_process', 'geometric_fbm',
	'FBMNoisyLinear', 'FractionalPositionalEmbedding',
	'estimate_hurst', 'FBMActionNoise',
	'get_default_device', 'get_default_dtype', 'set_seed'
]
