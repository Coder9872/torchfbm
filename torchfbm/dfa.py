import torch
import numpy as np


def dfa(
    x: torch.Tensor, scales: list = None, order: int = 1, return_alpha: bool = True
):
    """
    GPU-Accelerated Detrended Fluctuation Analysis (DFA).
    
    Based on Peng et al. (1994).
    Measures scaling: F(s) ~ s^alpha.

    Complexity: O(N) parallelized.
    Method: Batched Least Squares detrending.

    Args:
        x: Time series tensor (Batch, Time) or (Time,)
        scales: List of window sizes (int). If None, generates logarithmic range.
        order: Polynomial order for detrending (1 = Linear/DFA1, 2 = Quadratic/DFA2).
        return_alpha: If True, performs log-log regression to return the scaling exponent.
                      If False, returns the raw fluctuation function F(s).
    """
    # 1. Preprocessing
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1, Time)

    device = x.device
    N = x.shape[1]

    # Integrate (Cumulative Sum) - The "Profile"
    # y_k = sum(x_i - mean)
    y = torch.cumsum(x - x.mean(dim=1, keepdim=True), dim=1)

    # Default Scales (Log-spaced)
    if scales is None:
        min_scale = order + 2
        max_scale = N // 4
        scales = np.unique(
            np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int)
        )

    fluctuations = []

    for s in scales:
        # 2. Windowing (The Vectorization Hack)
        # We perform "strided" views or simple truncation to reshape into batches
        n_segments = N // s
        limit = n_segments * s

        # Reshape y into (Batch, Segments, Scale)
        # We discard the tail data for speed (standard DFA practice)
        y_truncated = y[:, :limit]
        y_segmented = y_truncated.view(x.shape[0], n_segments, s)

        # 3. Batched Detrending (The Speedup)
        # We want to fit y = X*beta + epsilon for EVERY segment at once.
        # X is the design matrix [1, t, t^2...] (Shape: s x order+1)
        # It is the SAME for every segment.

        t = torch.arange(s, device=device, dtype=x.dtype)
        # Create Vander matrix: [t^0, t^1, ... t^order]
        X = torch.stack([t**k for k in range(order + 1)], dim=1)  # (s, order+1)

        # Compute Pseudo-Inverse ONCE: (X^T X)^-1 X^T
        # Shape: (order+1, s)
        try:
            # fast least squares operator
            coeffs_op = torch.linalg.pinv(X)
        except:
            # Fallback for stability
            coeffs_op = torch.linalg.inv(X.T @ X) @ X.T

        # Calculate Trends: trend = X * beta
        # beta = coeffs_op @ y_segment^T
        # But we have batches.
        # y_segmented: (Batch, Segments, s)
        # coeffs_op: (order+1, s)

        # beta shape: (Batch, Segments, order+1)
        beta = torch.matmul(y_segmented, coeffs_op.t())

        # trend shape: (Batch, Segments, s)
        # trend = beta @ X.T
        trend = torch.matmul(beta, X.t())

        # 4. Fluctuation (RMS)
        # Residuals: y - trend
        rms = torch.sqrt(torch.mean((y_segmented - trend) ** 2, dim=2))

        # Average over all segments to get F(s)
        F_s = torch.mean(rms, dim=1)  # (Batch,)
        fluctuations.append(F_s)

    # Stack fluctuations: (Batch, Num_Scales)
    F = torch.stack(fluctuations, dim=1)

    if not return_alpha:
        return F, scales

    # 5. Log-Log Regression to find Alpha
    # log(F) = alpha * log(s) + C
    log_F = torch.log(F)
    log_scales = torch.log(torch.tensor(scales, device=device, dtype=x.dtype))

    # Simple Batched Linear Regression
    # Slope = Cov(x, y) / Var(x)
    S_xx = torch.var(log_scales, unbiased=False)
    mean_x = torch.mean(log_scales)
    mean_y = torch.mean(log_F, dim=1, keepdim=True)

    # Broadcast subtraction
    S_xy = torch.mean((log_scales - mean_x) * (log_F - mean_y), dim=1)

    alpha = S_xy / S_xx
    return alpha
