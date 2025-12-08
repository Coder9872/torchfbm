import torch
from typing import Optional
from .generators import generate_davies_harte, generate_cholesky


class FBMActionNoise:
    def __init__(self, mean, sigma, H=0.5, size=(1,), buffer_size=10000, method='davies_harte', device='cpu', return_torch=False, dtype=torch.float32, seed: Optional[int] = None):
        self._mu = mean
        self._sigma = sigma
        self._H = max(0.01, min(H, 0.99))
        self._size = size
        self._buffer_size = buffer_size
        self._method = method
        self._device = device
        self._return_torch = return_torch
        self._dtype = dtype
        self._seed = seed
        self.reset()

    def reset(self):
        gen_func = generate_cholesky if self._method == 'cholesky' else generate_davies_harte
        fgn = gen_func(self._buffer_size, self._H, size=self._size, device=self._device, dtype=self._dtype, seed=self._seed)
        self._noise_buffer_t = fgn
        # Keep computations in torch; avoid NumPy conversion
        self._step = 0

    def __call__(self):
        if self._step >= self._buffer_size:
            self.reset()
        noise_t = self._noise_buffer_t[..., self._step]
        self._step += 1
        base_mu = torch.as_tensor(self._mu, dtype=self._dtype, device=self._device)
        base_sigma = torch.as_tensor(self._sigma, dtype=self._dtype, device=self._device)
        out = base_mu + base_sigma * noise_t
        return out if self._return_torch else out.cpu()