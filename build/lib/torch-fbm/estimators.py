import torch

def estimate_hurst(x: torch.Tensor, min_lag: int = 2, max_lag: int = 20, assume_path: bool = True) -> torch.Tensor:
    """
    Estimates the Hurst Exponent H of a time series x using the 
    Aggregated Variance Method (Vectorized & Differentiable).
    
    Args:
        x: Time series tensor of shape (Batch, Time)
        min_lag, max_lag: Range of scales to check scaling laws.
        assume_path: If False, treats input as fGn and integrates to path.
        
    Returns:
        H: Estimated Hurst parameter for each batch element.
    """
    # Ensure x is variance-normalized for stability
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
    
    # Calculate cumulative sum (the integral of the path)
    # If input is noise (fGn), we integrate to get profile. 
    # If input is already path (fBm), we skip. 
    # Here assuming input is the Path (fBm) unless assume_path=False.
    if not assume_path:
        x = torch.cumsum(x, dim=-1)
    
    lags = torch.arange(min_lag, max_lag, device=x.device)
    variances = []
    
    for lag in lags:
        # Vectorized aggregation: X(t+tau) - X(t)
        # This computes the variance of increments at scale 'lag'
        # Var(|dX_tau|) ~ tau^(2H)
        if x.size(-1) <= lag:
            break
        increments = x[..., lag:] - x[..., :-lag]
        var = increments.var(dim=-1)
        variances.append(var)
        
    # Stack variances: Shape (Num_Lags, Batch)
    variances = torch.stack(variances, dim=0)
    
    # Linear Regression in Log-Log space
    # log(Var) = 2H * log(lag) + C
    y = torch.log(variances + 1e-8)
    X = torch.log(lags[:variances.size(0)].float()).unsqueeze(1).expand(-1, x.shape[0]) # (Num_Lags, Batch)
    
    # Simple Least Squares to find slope
    # Slope = Cov(X, Y) / Var(X)
    X_mean = X.mean(dim=0)
    y_mean = y.mean(dim=0)
    
    numerator = ((X - X_mean) * (y - y_mean)).sum(dim=0)
    denominator = ((X - X_mean) ** 2).sum(dim=0)
    
    slope = numerator / (denominator + 1e-8)
    
    # Slope = 2H, so H = Slope / 2
    H_est = slope / 2.0
    
    return torch.clamp(H_est, 0.01, 0.99)