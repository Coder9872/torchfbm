import torch
from .generators import _autocovariance

class CachedFGNGenerator:
    """
    Generates Fractional Gaussian Noise sequentially (point-by-point).
    Complexity: O(N^2) instead of O(N^3).
    
    Use Case: Real-time environments (RL) or Streaming Data.
    """
    def __init__(self, H: float, device='cpu'):
        self.H = H
        self.device = torch.device(device)
        self.n = 0
        
        # The Cholesky Factor (Lower Triangular)
        # We start empty.
        self.L = torch.zeros(0, 0, device=self.device)
        self.history = torch.zeros(0, device=self.device)
        
    def step(self) -> torch.Tensor:
        """
        Generates the next point in the fGn sequence.
        Returns a scalar (or batch if vectorized, keeping scalar for clarity here).
        """
        new_idx = self.n
        
        # 1. Calculate the new column of covariance 'v'
        # Cov(X_new, X_old) depends on lag |i - j|
        # lags = [n, n-1, ..., 1]
        if new_idx == 0:
            # First point is just N(0, 1) since variance of fGn is 1
            self.L = torch.ones(1, 1, device=self.device)
            noise = torch.randn(1, device=self.device)
            val = noise
        else:
            # Calculate covariance vector v = [gamma(n), gamma(n-1), ..., gamma(1)]
            # Note: _autocovariance returns [gamma(0), gamma(1), ...]
            # We want gamma(1) to gamma(n)
            # gamma(0) is the diagonal 'c' which is 1.0 for fGn
            
            full_gamma = _autocovariance(self.H, new_idx + 1, self.device)
            v = full_gamma[1:].flip(0) # [gamma(n), ..., gamma(1)]
            c = full_gamma[0] # gamma(0) = 1.0
            
            # 2. Solve L_n * w = v for w
            # Since L is lower triangular, we use solve_triangular
            # v needs to be (N, 1)
            v = v.unsqueeze(1)
            w = torch.linalg.solve_triangular(self.L, v, upper=False)
            
            # 3. Calculate delta
            # delta = sqrt(c - w^T * w)
            w_norm_sq = torch.dot(w.flatten(), w.flatten())
            delta = torch.sqrt(c - w_norm_sq)
            
            # 4. Sample new white noise
            z = torch.randn(1, device=self.device)
            
            # 5. Result = w^T * history_noise + delta * z
            # Wait, standard Cholesky generation is X = L * Z
            # X_new = w^T * Z_old + delta * z_new
            # We don't need Z_old if we just maintain X? 
            # Actually, standard generation X = L @ Z. 
            # We need to store Z (the white noise history), not X (the correlated history).
            
            # We need to store the white noise 'Z' history to generate next step?
            # Yes. X = L*Z. 
            # X_{n+1} (last row) = w^T * Z_{0:n} + delta * z_{n+1}
            
            # Let's verify storage
            if not hasattr(self, 'z_history'):
                self.z_history = torch.zeros(0, device=self.device)
                
            val = torch.dot(w.flatten(), self.z_history) + delta * z
            
            # 6. Update L matrix (Append row/col)
            # New L is [[L, 0], [w.T, delta]]
            # This is the slow part (memory copy). 
            # Optimization: Pre-allocate L and double size when full?
            # For simplicity: Concatenate
            
            # Row to add: [w^T, delta]
            bottom_row = torch.cat([w.flatten(), delta.unsqueeze(0)])
            
            # Pad L with zeros column
            zeros_col = torch.zeros(self.n, 1, device=self.device)
            L_padded = torch.cat([self.L, zeros_col], dim=1)
            
            self.L = torch.vstack([L_padded, bottom_row])
            self.z_history = torch.cat([self.z_history, z])
            
        self.n += 1
        return val