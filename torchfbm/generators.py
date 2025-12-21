import torch
import numpy
def _autocovariance(H: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(0, n, device=device, dtype=dtype)
    return 0.5 * (torch.abs(k + 1)**(2 * H) - 2 * torch.abs(k)**(2 * H) + torch.abs(k - 1)**(2 * H))

def generate_cholesky(
    n: int, H: float, size: tuple = (1,), 
    device='cpu', dtype=torch.float32, seed=None, return_numpy: bool = False
) -> torch.Tensor:
    """Exact O(N^3) method."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    gamma = _autocovariance(H, n, device, dtype)
    idx = torch.arange(n, device=device)
    lhs = idx.unsqueeze(0)
    rhs = idx.unsqueeze(1)
    distance_matrix = torch.abs(lhs - rhs)
    Sigma = gamma[distance_matrix]
    
    jitter = 1e-6 * torch.eye(n, device=device, dtype=dtype)
    try:
        L = torch.linalg.cholesky(Sigma + jitter)
    except RuntimeError:
        L = torch.linalg.cholesky(Sigma + jitter * 10) 

    noise = torch.randn(*size, n, device=device, dtype=dtype, generator=generator)
    result = torch.matmul(noise, L.t())
    return result.cpu().numpy() if return_numpy else result

def generate_davies_harte(
    n: int, H: float, size: tuple = (1,), 
    device='cpu', dtype=torch.float32, seed=None, return_numpy: bool = False
) -> torch.Tensor:
    """Fast O(N log N) method."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    gamma = _autocovariance(H, n, device, dtype)
    
    # Davies-Harte embedding
    row = torch.cat([gamma, gamma[1:-1].flip(0)])
    M = row.shape[0]
    
    # FFT (Real to Complex)
    lambdas = torch.fft.fft(row).real
    lambdas = torch.clamp(lambdas, min=0.0)
    
    # Generate Complex White Noise with specific generator/dtype
    rng_real = torch.randn(*size, M, device=device, dtype=dtype, generator=generator)
    rng_imag = torch.randn(*size, M, device=device, dtype=dtype, generator=generator)
    complex_noise = torch.complex(rng_real, rng_imag)
    
    scale = torch.sqrt(lambdas / M)
    fft_noise = complex_noise * scale
    
    simulation = torch.fft.ifft(fft_noise) * M
    result = simulation.real[..., :n]
    return result.cpu().numpy() if return_numpy else result

def fbm(
    n: int, H: float, size: tuple = (1,), 
    method='davies_harte', device='cpu', dtype=torch.float32, seed=None, return_numpy: bool = False
):
    """
    Main Entry Point.
    Standardized to match torch.randn API (size, device, dtype, generator).
    """
    H = max(0.01, min(H, 0.99))
    
    if method == 'cholesky':
        func = generate_cholesky
    else:
        func = generate_davies_harte
        
    fgn = func(n, H, size, device=device, dtype=dtype, seed=seed, return_numpy=False)
    
    zeros = torch.zeros(*size, 1, device=device, dtype=dtype)
    result = torch.cat([zeros, torch.cumsum(fgn, dim=-1)], dim=-1)
    return result.cpu().numpy() if return_numpy else result