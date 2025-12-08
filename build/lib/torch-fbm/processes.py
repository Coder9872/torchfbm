import torch
from .generators import generate_davies_harte, generate_cholesky, fbm

def fractional_ou_process(n: int, H: float, theta: float = 0.5, mu: float = 0.0, sigma: float = 1.0, dt: float = 0.01, size: tuple = (1,), method: str = 'davies_harte', device: str = 'cpu', dtype: torch.dtype = torch.float32):
    """
    Simulates a Fractional Ornstein-Uhlenbeck (fOU) process.
    dX_t = theta * (mu - X_t) * dt + sigma * dW_H(t)
    
    Args:
        method: 'davies_harte' (fast) or 'cholesky' (exact)
    """
    # Safety Clamp
    H = max(0.01, min(H, 0.99))

    # Select Generator
    if method == 'cholesky':
        gen_func = generate_cholesky
    else:
        gen_func = generate_davies_harte

    # 1. Generate Fractional Gaussian Noise (Increments)
    fgn = gen_func(n, H, size, device=device, dtype=dtype)
    
    # 2. Scale noise
    # Standard scaling for fBm increments is dt^H
    noise_term = sigma * fgn * (dt ** H)
    
    # 3. Euler-Maruyama Integration
    x = torch.zeros(*size, n + 1, device=device, dtype=dtype)
    x[..., 0] = mu # Start at mean
    
    drift_factor = 1 - theta * dt
    drift_constant = theta * mu * dt
    
    # Loop over time (cannot be easily vectorized due to recursive dependency)
    for t in range(n):
        x[..., t+1] = x[..., t] * drift_factor + drift_constant + noise_term[..., t]
        
    return x

def geometric_fbm(
    n: int, 
    H: float, 
    mu: float = 0.05, 
    sigma: float = 0.2, 
    t_max: float = 1.0,
    s0: float = 100.0,
    size: tuple = (1,), 
    method: str = 'davies_harte',
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
):
    """
    Simulates Geometric Fractional Brownian Motion (Asset Prices).
    S_t = S_0 * exp( (mu - 0.5*sigma^2)t + sigma * B_H(t) )
    """
    device = torch.device(device)
    
    # Time grid
    t = torch.linspace(0, t_max, n + 1, device=device, dtype=dtype).expand(*size, n + 1)
    
    # Generate standard fBm path B_H(t)
    # We use the fbm() wrapper which handles H-clamping and method selection
    # Note: fbm() returns shape (..., n+1) starting at 0
    fbm_path = fbm(n, H, size=size, method=method, device=device, dtype=dtype)
    
    # Scale time horizon:
    # The fbm() generator assumes T=n (unit steps). We need to scale to T=t_max.
    # Scaling law: B(at) ~ a^H * B(t)
    # So we scale by (dt)^H is handled implicitly if we view steps as dt? 
    # Actually, easier to just rescale the final path:
    # The generated path reaches 'n'. We want it to reach 't_max'.
    # Rescaling factor: (t_max / n)^H ?? 
    # Let's stick to standard definition: B_H(t) has variance t^(2H).
    # Our generated fbm_path has variance n^(2H) at the end.
    # We want variance t_max^(2H).
    
    scale_factor = (t_max / n) ** H
    fbm_path = fbm_path * scale_factor
    
    # Geometric formula
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * fbm_path
    
    log_returns = drift + diffusion
    
    return s0 * torch.exp(log_returns)