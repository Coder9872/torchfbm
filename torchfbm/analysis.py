import torch
import torch.fft
import math


def covariance_matrix(
    n: int, H: float, device="cpu", return_numpy: bool = False
) -> torch.Tensor:
    """Returns the exact TxT Autocovariance Matrix for fGn."""
    from .generators import _autocovariance

    gamma = _autocovariance(H, n, torch.device(device), dtype=torch.float32)

    # Toeplitz construction
    idx = torch.arange(n, device=device)
    distance_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    result = gamma[distance_matrix]
    return result.cpu().numpy() if return_numpy else result


def plot_acf(x: torch.Tensor, max_lag: int = 100, title="Autocorrelation"):
    """
    Plots the ACF of a time series.
    Requires matplotlib (install separately).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed.")
        return

    # Calculate ACF via FFT (Faster than loops)
    n = x.shape[-1]
    x_centered = x - x.mean()
    # Pad to avoid circular correlation
    x_padded = torch.nn.functional.pad(x_centered, (0, n))
    fft = torch.fft.fft(x_padded)
    power = fft * fft.conj()
    acf = torch.fft.ifft(power).real
    acf = acf[:max_lag]
    acf = acf / acf[0]  # Normalize

    # Plot
    acf_np = acf.detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(acf_np)), acf_np, width=0.5)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.show()


def spectral_scaling_factor(
    f: torch.Tensor, H: float, return_numpy: bool = False
) -> torch.Tensor:
    """
    Returns the spectral scaling factor for spectral synthesis of fBm.
    
    Based on the 1/f^beta power spectral density law.
    Scaling: A(f) ~ 1/f^(H + 0.5)

    Args:
        f: Frequencies.
        H: Hurst exponent.
    """
    beta = 2 * H + 1
    # Avoid div by zero at f=0
    safe_f = torch.where(f == 0, torch.ones_like(f), f)
    scaling = 1.0 / (torch.abs(safe_f) ** (beta / 2.0))
    scaling[f == 0] = 0  # DC component usually 0 for Brownian
    return scaling.cpu().numpy() if return_numpy else scaling
