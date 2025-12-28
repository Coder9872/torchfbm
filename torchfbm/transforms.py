import torch
import torch.fft


def fractional_diff(
    x: torch.Tensor, d: float, dim: int = -1, return_numpy: bool = False
) -> torch.Tensor:
    """
    Computes the Fractional Derivative (or Integral if d < 0) using FFT.
    This preserves memory better than standard differencing.

    Args:
        x: Input tensor (Batch, Time)
        d: Fractional order (e.g., 0.4).
           d=0 is identity. d=1 is standard diff.
    """
    # Based on the Jensen/Whitcher definition via Frequency Domain
    # (1 - L)^d where L is lag operator.

    dim = dim if dim >= 0 else x.dim() + dim
    if dim < 0 or dim >= x.dim():
        raise ValueError(f"Invalid dim {dim} for input with {x.dim()} dims")

    n = x.shape[dim]
    device = x.device

    if n == 0:
        raise ValueError(
            "Cannot compute fractional difference along an empty dimension"
        )

    real_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
    if real_dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise TypeError(f"Unsupported dtype {real_dtype} for fractional_diff")

    # 1. Compute Weights via FFT formulation
    # The transfer function is (1 - exp(-i 2pi k / n))^d
    # We compute this in Fourier space directly for speed.

    # Frequencies
    freq_dtype = torch.promote_types(real_dtype, torch.float32)
    k = torch.arange(n, device=device, dtype=freq_dtype)
    # The operator in frequency domain: (1 - e^(-i omega))^d
    omega = 2 * torch.tensor(torch.pi, device=device, dtype=freq_dtype) * k / n

    # Transfer function
    transfer = (1 - torch.exp(-1j * omega)) ** d

    # Guard zero-frequency when d < 0 (fractional integral)
    if d < 0:
        transfer = transfer.clone()
        transfer[0] = torch.tensor(1.0, device=device, dtype=transfer.dtype)

    # Reshape transfer for broadcasting along arbitrary dim
    view_shape = [1] * x.dim()
    view_shape[dim] = n
    transfer = transfer.reshape(view_shape)

    # 2. Apply via FFT Convolution
    x_fft = torch.fft.fft(x, dim=dim)
    transfer = transfer.to(device=device, dtype=x_fft.dtype)
    diff_fft = x_fft * transfer
    x_diff = torch.fft.ifft(diff_fft, dim=dim).real

    return x_diff.cpu().numpy() if return_numpy else x_diff


def fractional_integrate(
    x: torch.Tensor, d: float, dim: int = -1, return_numpy: bool = False
) -> torch.Tensor:
    """Inverse of Fractional Diff. Makes a series 'smoother'."""
    return fractional_diff(x, -d, dim=dim, return_numpy=return_numpy)
