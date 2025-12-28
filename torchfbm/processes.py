import torch
from .generators import generate_davies_harte, generate_cholesky, fbm


def fractional_ou_process(
    n: int,
    H: float,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.01,
    size: tuple = (1,),
    method: str = "davies_harte",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = False,
):
    """
    Simulates a Fractional Ornstein-Uhlenbeck (fOU) process.
    
    Based on Cheridito et al. (2003).
    Equation: dX_t = theta * (mu - X_t) * dt + sigma * dB_H(t)

    Args:
        method: 'davies_harte' (fast) or 'cholesky' (exact)
    """
    # Safety Clamp
    H = max(0.01, min(H, 0.99))

    # Select Generator
    if method == "cholesky":
        gen_func = generate_cholesky
    else:
        gen_func = generate_davies_harte

    # 1. Generate Fractional Gaussian Noise (Increments)
    fgn = gen_func(n, H, size, device=device, dtype=dtype, return_numpy=False)

    # 2. Scale noise
    # Standard scaling for fBm increments is dt^H
    noise_term = sigma * fgn * (dt**H)

    # 3. Euler-Maruyama Integration
    x = torch.zeros(*size, n + 1, device=device, dtype=dtype)
    x[..., 0] = mu  # Start at mean

    drift_factor = 1 - theta * dt
    drift_constant = theta * mu * dt

    # Loop over time (cannot be easily vectorized due to recursive dependency)
    for t in range(n):
        x[..., t + 1] = x[..., t] * drift_factor + drift_constant + noise_term[..., t]

    return x.cpu().numpy() if return_numpy else x


def geometric_fbm(
    n: int,
    H: float,
    mu: float = 0.05,
    sigma: float = 0.2,
    t_max: float = 1.0,
    s0: float = 100.0,
    size: tuple = (1,),
    method: str = "davies_harte",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = False,
):
    """
    Simulates Geometric Fractional Brownian Motion (GFBm).
    
    Standard model for asset prices with long-range dependence.
    Equation: S_t = S_0 * exp( (mu - 0.5*sigma^2)t + sigma * B_H(t) )
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    device = torch.device(device)

    # Time grid
    t = torch.linspace(0, t_max, n + 1, device=device, dtype=dtype).expand(*size, n + 1)

    # Generate standard fBm path B_H(t)
    # We use the fbm() wrapper which handles H-clamping and method selection
    # Note: fbm() returns shape (..., n+1) starting at 0
    fbm_path = fbm(
        n, H, size=size, method=method, device=device, dtype=dtype, return_numpy=False
    )

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

    result = s0 * torch.exp(log_returns)
    return result.cpu().numpy() if return_numpy else result


import torch
from .generators import fbm  # Ensure fbm is imported


@torch.jit.script
def _apply_reflection(
    path: torch.Tensor, increments: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
    """
    JIT-compiled loop to apply reflection boundaries efficiently.
    X_{t+1} = X_t + dW_t
    If X_{t+1} > Upper: Reflect down
    If X_{t+1} < Lower: Reflect up
    """
    # Create output tensor
    # increments shape: (..., T)
    # path shape: (..., T+1)

    # Iterate through time
    # Note: We assume the first point is already set (usually 0 or S0)
    T = increments.shape[-1]

    # We flatten batch dims for the loop to make it simple, then reshape back?
    # JIT handles arbitrary shapes fairly well, but simple loops are safest.
    # Let's just iterate over time dimension T

    for t in range(T):
        # Propose next step
        dx = increments[..., t]
        current_x = path[..., t]
        next_x = current_x + dx

        # Check Upper Bound
        # If next_x > upper: overshoot is (next_x - upper)
        # reflected = upper - overshoot = upper - (next_x - upper) = 2*upper - next_x

        # We use torch.where to handle batch logic without python branching
        over_upper = next_x > upper
        next_x = torch.where(over_upper, 2 * upper - next_x, next_x)

        # Check Lower Bound
        # If next_x < lower: undershoot is (lower - next_x)
        # reflected = lower + undershoot = lower + (lower - next_x) = 2*lower - next_x
        under_lower = next_x < lower
        next_x = torch.where(under_lower, 2 * lower - next_x, next_x)

        # Store
        path[..., t + 1] = next_x

    return path


def reflected_fbm(
    n: int,
    H: float,
    lower: float = -1.0,
    upper: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    t_max: float = 1.0,
    start_val: float = 0.0,
    size: tuple = (1,),
    method: str = "davies_harte",
    device: str = "cpu",
    return_numpy: bool = False,
):
    """
    Simulates Reflected Fractional Brownian Motion (Bounded fBm) in [lower, upper].
    Uses the Skorokhod reflection map via a JIT-compiled solver.

    Useful for:
    - Modeling pegged currencies (Target Zones).
    - Particles in a confined box (Physics).
    - Mean-reverting assets with hard support/resistance.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    device = torch.device(device)
    # 1. Generate fGn (Increments)
    # We use fbm() to handle method/clamping, but we need the *steps*, not the path.
    # So we call the generator directly or diff the fbm path.
    # Let's diff the fbm path for consistency with the geometric_fbm scaling logic.

    # Generate unbounded path first to get correctly scaled increments
    unbounded_path = fbm(n, H, size, method=method, device=device, return_numpy=False)

    # Scale to t_max and sigma
    scale_factor = sigma * (t_max / n) ** H
    unbounded_path = unbounded_path * scale_factor

    # Add drift (mu * dt)
    dt = t_max / n
    t_grid = torch.linspace(0, t_max, n + 1, device=device).expand(*size, n + 1)
    drift = mu * t_grid

    # Combine to get the proposed increments with drift
    total_unbounded = unbounded_path + drift

    # Calculate increments (dX)
    increments = total_unbounded[..., 1:] - total_unbounded[..., :-1]

    # 2. Prepare container
    reflected_path = torch.zeros_like(total_unbounded)
    reflected_path[..., 0] = start_val

    # 3. Run JIT Reflection
    # Ensure bounds are floats for JIT
    reflected_path = _apply_reflection(
        reflected_path, increments, float(lower), float(upper)
    )

    return reflected_path.cpu().numpy() if return_numpy else reflected_path


def fractional_brownian_bridge(
    n: int,
    H: float,
    start_val: float = 0.0,
    end_val: float = 0.0,
    t_max: float = 1.0,
    sigma: float = 1.0,
    size: tuple = (1,),
    method: str = "davies_harte",
    device: str = "cpu",
    return_numpy: bool = False,
):
    """
    Simulates a Fractional Brownian Bridge.
    A rough path that is conditioned to start at `start_val` and end at `end_val`.

    Uses the "Fast Pinning" method:
    X_bridge(t) = X_free(t) - (t/T) * (X_free(T) - target_displacement)
    """
    device = torch.device(device)

    # 1. Generate a FREE unconditioned path starting at 0
    # We use fbm() to handle H-clamping and method selection
    # shape: (..., n+1)
    free_path = fbm(n, H, size=size, method=method, device=device, return_numpy=False)

    # 2. Scale the free path to physical time/sigma
    # Scale factor for variance over time T is T^(2H)
    # The generator gives us unit step variance.
    scale_factor = sigma * (t_max / n) ** H
    free_path = free_path * scale_factor

    # 3. Create the Time Grid
    # Shape: (1, ..., n+1) to broadcast correctly
    t_grid = torch.linspace(0, t_max, n + 1, device=device)
    # Reshape for broadcasting against 'size'
    # If size=(Batch, ), free_path is (Batch, n+1).
    # We need t_grid to be (1, n+1) compatible.
    for _ in range(len(size)):
        t_grid = t_grid.unsqueeze(0)

    # 4. Calculate the Bridge "Drift"
    # We need to subtract the error at the end.
    # The free path ends at X_T. We want it to end at (end_val - start_val).
    # Current endpoint error = free_path[..., -1]

    current_end = free_path[..., -1:]  # Keep dims for broadcast
    target_displacement = end_val - start_val

    # The correction term is linear interpolation of the error
    # Correction(t) = (t / T) * (current_end - target_displacement)
    correction = (t_grid / t_max) * (current_end - target_displacement)

    # 5. Apply correction and shift start
    bridge = free_path - correction + start_val

    return bridge.cpu().numpy() if return_numpy else bridge


def multifractal_random_walk(n, H, lambda_sq=0.02, device="cpu"):
    """
    Simulates MRW.
    lambda_sq: Intermittency coefficient (higher = more flash crashes).
    """
    # 1. Generate fGn for the volatility cone (omega)
    # The correlation of omega is logarithmic
    # This is complex to do perfectly, but a proxy is:
    omega = generate_davies_harte(n, H, device=device)

    # 2. Stochastic Volatility
    sigma = torch.exp(lambda_sq * omega)

    # 3. The Walk
    noise = torch.randn(n, device=device)
    mrw = torch.cumsum(sigma * noise, dim=-1)
    return mrw
