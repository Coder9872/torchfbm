import torch
from .estimators import estimate_hurst


class HurstRegularizationLoss(torch.nn.Module):
    """
    Penalizes deviations from a target Hurst exponent.
    L = lambda * (H_est(y_pred) - H_target)^2

    Usage:
    loss = mse_loss(y_pred, y_true) + 0.1 * hurst_reg(y_pred)
    """

    def __init__(self, target_H: float = 0.5):
        super().__init__()
        self.target_H = target_H

    def forward(self, x):
        # Estimate H of the batch
        # x shape: (Batch, Time)
        h_est = estimate_hurst(x)  # Returns (Batch,)

        # Mean Squared Error from target
        return torch.mean((h_est - self.target_H) ** 2)


import torch


class SpectralConsistencyLoss(torch.nn.Module):
    """
    Penalizes deviations from the target 1/f^beta spectral slope.
    Ensures generated samples have correct multi-scale roughness.
    Target: S(f) ~ 1/f^(2H+1)
    """

    def __init__(self, target_beta: float):
        super().__init__()
        self.target_beta = target_beta  # beta = 2H + 1 for fBm

    def forward(self, x: torch.Tensor):
        # x: (Batch, Length)
        n = x.shape[-1]

        # Compute PSD
        fft = torch.fft.rfft(x, dim=-1)
        psd = torch.abs(fft) ** 2
        freqs = torch.fft.rfftfreq(n, d=1.0).to(x.device)

        # Target PSD = C / f^beta
        # We work in Log-Log space to fit the slope
        # Ignore DC component (freq=0)
        log_f = torch.log(freqs[1:] + 1e-8)
        log_psd = torch.log(psd[..., 1:] + 1e-8)

        # Calculate slope of generated data
        # (Simple linear regression slope)
        mean_x = log_f.mean()
        mean_y = log_psd.mean(dim=-1, keepdim=True)

        num = ((log_f - mean_x) * (log_psd - mean_y)).sum(dim=-1)
        den = ((log_f - mean_x) ** 2).sum()

        estimated_beta = -num / (den + 1e-8)

        # Loss: Distance from target slope (beta)
        return torch.mean((estimated_beta - self.target_beta) ** 2)
