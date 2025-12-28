import torch
from .generators import _autocovariance


class CachedFGNGenerator:
    """
    Generates Fractional Gaussian Noise sequentially (point-by-point).
    Complexity: O(N^2) instead of O(N^3).

    Use Case: Real-time environments (RL) or Streaming Data.
    """

    def __init__(self, H: float, device="cpu", dtype=torch.float32):
        self.H = H
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = 0

        # The Cholesky Factor (Lower Triangular)
        self.L = torch.zeros(0, 0, device=self.device, dtype=self.dtype)
        # History of uncorrelated white noise (Z)
        self.z_history = torch.zeros(0, device=self.device, dtype=self.dtype)

    def step(self) -> torch.Tensor:
        """
        Generates the next point in the fGn sequence.
        """
        new_idx = self.n

        # 1. Generate new white noise sample (z_new)
        # Ensure it is 1D [1] so concatenation works later
        z = torch.randn(1, device=self.device, dtype=self.dtype)

        if new_idx == 0:
            # First point initialization
            # Variance of fGn is 1.0, so L is [[1.0]]
            self.L = torch.ones(1, 1, device=self.device, dtype=self.dtype)
            val = z

            # Update history
            self.z_history = torch.cat([self.z_history, z])

        else:
            # 2. Calculate the new column of covariance 'v'
            full_gamma = _autocovariance(self.H, new_idx + 1, self.device, self.dtype)
            v = full_gamma[1:].flip(0)  # [gamma(n), ..., gamma(1)]
            c = full_gamma[0]  # gamma(0) = 1.0

            # 3. Solve L_n * w = v for w (Forward Substitution)
            # v needs to be (N, 1)
            v = v.unsqueeze(1)
            w = torch.linalg.solve_triangular(self.L, v, upper=False)

            # 4. Calculate delta
            w_flat = w.flatten()
            w_norm_sq = torch.dot(w_flat, w_flat)

            # Numerical clamp for safety
            delta_sq = c - w_norm_sq
            delta = torch.sqrt(torch.clamp(delta_sq, min=1e-8))  # 0-dim Scalar

            # 5. Calculate Result
            # X_{n+1} = w^T * Z_{0:n} + delta * z_{n+1}
            # w_flat (N), z_history (N) -> dot is 0-dim Scalar
            # delta (0-dim), z (1-dim) -> product is 1-dim [1]
            # scalar + [1] -> [1]
            val = torch.dot(w_flat, self.z_history) + delta * z

            # 6. Update State
            self.z_history = torch.cat([self.z_history, z])

            # Update L matrix: [[L, 0], [w^T, delta]]
            # Pad L
            zeros_col = torch.zeros(self.n, 1, device=self.device, dtype=self.dtype)
            L_padded = torch.cat([self.L, zeros_col], dim=1)

            # Prepare bottom row
            # FIX: delta is 0-dim, must be 1-dim to cat with w_flat
            bottom_row = torch.cat([w_flat, delta.unsqueeze(0)])

            # Stack
            self.L = torch.vstack([L_padded, bottom_row.unsqueeze(0)])

        self.n += 1
        return val
