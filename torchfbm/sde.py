import torch
import torch.nn as nn
from .generators import fbm, generate_davies_harte


class NeuralFSDE(nn.Module):
    """
    Solves a Fractional Stochastic Differential Equation (fSDE) using a Neural Network.
    
    Equation: dX_t = mu(X,t)dt + sigma(X,t)dB^H_t
    Uses Euler-Maruyama discretization for fSDEs.

    The Hurst parameter H can be fixed or learnable.
    """

    def __init__(
        self,
        state_size,
        drift_net,
        diffusion_net,
        H_init=0.5,
        learnable_H=False,
        t_max=1.0,
    ):
        super().__init__()
        self.state_size = state_size
        self.drift_net = drift_net  # Neural Net for Drift (mu)
        self.diffusion_net = diffusion_net  # Neural Net for Diffusion (sigma)
        self.t_max = t_max

        # Make H learnable?
        # We use a sigmoid wrapper to keep H in (0, 1) during optimization
        raw_h = torch.tensor(H_init).logit()  # Inverse sigmoid
        if learnable_H:
            self.raw_h = nn.Parameter(raw_h)
        else:
            self.register_buffer("raw_h", raw_h)

    @property
    def H(self):
        # Clamp to safe range [0.01, 0.99]
        return torch.sigmoid(self.raw_h) * 0.98 + 0.01

    def forward(self, x0, n_steps, method="davies_harte"):
        batch_size = x0.shape[0]
        dt = self.t_max / n_steps
        device = x0.device

        # 1. Generate Driver (Fractional Brownian Noise)
        # We need the INCREMENTS (d B^H_t)
        # Note: We generate fresh noise every forward pass to allow stochastic training
        # IMPORTANT: This generation path is fully differentiable w.r.t. self.H

        # Scale: fGn is unit variance. Scale by dt^H for physical time step.
        h_curr = self.H
        fgn = generate_davies_harte(
            n_steps, h_curr, size=(batch_size, self.state_size), device=device
        )
        noise_increments = fgn * (dt**h_curr)

        # 2. Euler-Maruyama Integration
        xt = x0
        trajectory = [x0]
        t = 0.0

        for i in range(n_steps):
            # Evaluate Neural Nets
            # Inputs: Current State + Time (optional, usually concatenated)
            # Simple implementation: pass state only
            drift = self.drift_net(xt) * dt
            diffusion = self.diffusion_net(xt)

            # Noise term: sigma * dB^H
            noise = diffusion * noise_increments[..., i]

            xt = xt + drift + noise
            trajectory.append(xt)
            t += dt

        return torch.stack(trajectory, dim=1)
