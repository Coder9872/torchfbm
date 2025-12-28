import torch
from .generators import generate_davies_harte


class FractionalNoiseAugmentation(torch.nn.Module):
    """
    Augments data by adding Fractional Gaussian Noise.
    Helps models become robust to long-term trends (H>0.5)
    or high-frequency roughness (H<0.5).
    """

    def __init__(self, H: float = 0.5, sigma: float = 0.01, p: float = 0.5):
        super().__init__()
        self.H = H
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            noise = generate_davies_harte(
                x.shape[-1], self.H, size=x.shape[:-1], device=x.device
            )
            return x + self.sigma * noise
        return x
