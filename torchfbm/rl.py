import numpy as np
import torch
from .generators import generate_davies_harte, generate_cholesky

class FBMActionNoise:
    """
    Fractional Gaussian Noise generator for RL Action Space Exploration.
    Compatible with Stable Baselines3 'ActionNoise' interface (if return_numpy=True).
    
    Args:
        return_numpy (bool): If False (default), returns PyTorch tensors (on device).
                             If True, returns NumPy arrays (SB3 compatible).
    """
    def __init__(self, mean, sigma, H=0.5, size=(1,), buffer_size=10000, 
                 method='davies_harte', device='cpu', return_numpy=False):
        self._mu = mean
        self._sigma = sigma
        self._H = H
        self._size = size
        self._buffer_size = buffer_size
        self._method = method
        self._device = device
        self._return_numpy = return_numpy
        
        self.reset()

    def reset(self):
        """Pre-generates a long buffer of fGn to sample from."""
        if self._method == 'cholesky':
            gen_func = generate_cholesky
        else:
            gen_func = generate_davies_harte

        # Generate on the requested device
        fgn = gen_func(
            self._buffer_size, 
            self._H, 
            size=self._size, 
            device=self._device
        )
        
        # Handle Output Type
        if self._return_numpy:
            try:
                self._noise_buffer = fgn.detach().cpu().numpy() # Convert to NumPy on CPU
            except RuntimeError as e:
                if "Numpy is not available" in str(e):
                    raise ImportError(
                        "NumPy conversion requested but NumPy is not properly installed or has compatibility issues. "
                        "Please install/upgrade numpy: pip install -U numpy"
                    ) from e
                raise
        else:
            self._noise_buffer = fgn # Keep as Tensor on Device
            
        self._step = 0
    
    def __call__(self):
        """Returns the noise for the current step."""
        if self._step >= self._buffer_size:
            self.reset()
            
        # Get noise step
        noise = self._noise_buffer[..., self._step]
        self._step += 1
        
        # Calculate result
        val = self._mu + self._sigma * noise
        
        # Ensure correct return type
        if self._return_numpy:
            # Force NumPy
            if isinstance(val, torch.Tensor):
                try:
                    return val.detach().cpu().numpy()
                except RuntimeError as e:
                    if "Numpy is not available" in str(e):
                        raise ImportError(
                            "NumPy conversion requested but NumPy is not properly installed or has compatibility issues. "
                            "Please install/upgrade numpy: pip install -U numpy"
                        ) from e
                    raise
            return np.asarray(val)
        else:
            # If val is somehow numpy/float, cast to tensor
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, device=self._device)
            return val
            
    def __repr__(self) -> str:
        return f"FBMActionNoise(mu={self._mu}, sigma={self._sigma}, H={self._H}, numpy={self._return_numpy})"