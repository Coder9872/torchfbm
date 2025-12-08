import torch
from typing import Optional

def _autocovariance(H: float, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    k = torch.arange(0, n, device=device, dtype=dtype)
    return 0.5 * (torch.abs(k + 1)**(2 * H) - 2 * torch.abs(k)**(2 * H) + torch.abs(k - 1)**(2 * H))

def generate_cholesky(n: int, H: float, size: tuple = (1,), device='cpu', dtype: torch.dtype = torch.float32, seed: Optional[int] = None) -> torch.Tensor:
    device = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    gamma = _autocovariance(H, n, device, dtype)
    idx = torch.arange(n, device=device)
    lhs = idx.unsqueeze(0)
    rhs = idx.unsqueeze(1)
    distance_matrix = torch.abs(lhs - rhs)
    Sigma = gamma[distance_matrix].to(dtype)
    jitter = torch.tensor(1e-6, dtype=dtype, device=device) * torch.eye(n, dtype=dtype, device=device)
    L = torch.linalg.cholesky(Sigma + jitter)
    noise = torch.randn(*size, n, device=device, dtype=dtype)
    return torch.matmul(noise, L.t())

def generate_davies_harte(n: int, H: float, size: tuple = (1,), device='cpu', dtype: torch.dtype = torch.float32, seed: Optional[int] = None) -> torch.Tensor:
    device = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    gamma = _autocovariance(H, n, device, dtype)
    row = torch.cat([gamma, gamma[1:-1].flip(0)])
    M = row.shape[0]
    lambdas = torch.fft.fft(row).real
    lambdas = torch.clamp(lambdas, min=torch.finfo(dtype).tiny)  # tiny floor
    rng_real = torch.randn(*size, M, device=device, dtype=dtype)
    rng_imag = torch.randn(*size, M, device=device, dtype=dtype)
    complex_noise = torch.complex(rng_real, rng_imag)
    scale = torch.sqrt(lambdas / M)
    fft_noise = complex_noise * scale
    simulation = torch.fft.ifft(fft_noise) * M
    return simulation.real[..., :n].to(dtype)

def fbm(n: int, H: float, size: tuple = (1,), method='davies_harte', device='cpu', dtype: torch.dtype = torch.float32, seed: Optional[int] = None):
    H = max(0.01, min(H, 0.99))
    if method == 'cholesky':
        fgn = generate_cholesky(n, H, size, device, dtype=dtype, seed=seed)
    else:
        fgn = generate_davies_harte(n, H, size, device, dtype=dtype, seed=seed)
    zeros = torch.zeros(*size, 1, device=device, dtype=dtype)
    return torch.cat([zeros, torch.cumsum(fgn, dim=-1)], dim=-1)