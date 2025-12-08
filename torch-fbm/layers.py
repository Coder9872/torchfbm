import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from .generators import generate_davies_harte, generate_cholesky

class FBMNoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, H=0.5, sigma_init=0.5, buffer_size=1000, method='davies_harte', device='cpu', dtype=torch.float32, seed: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.H = H
        self.buffer_size = buffer_size
        self.method = method  # Store preference
        self.step_counter = 0
        self._device = torch.device(device)
        self._dtype = dtype
        self._seed = seed
        
        # ... (Parameters definitions same as before) ...
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=self._device, dtype=self._dtype))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, device=self._device, dtype=self._dtype))
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=self._device, dtype=self._dtype))
        self.bias_sigma = nn.Parameter(torch.empty(out_features, device=self._device, dtype=self._dtype))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features, device=self._device, dtype=self._dtype))
        self.register_buffer('bias_epsilon', torch.empty(out_features, device=self._device, dtype=self._dtype))
        self.register_buffer('noise_stream_in', torch.empty(in_features, buffer_size, device=self._device, dtype=self._dtype))
        self.register_buffer('noise_stream_out', torch.empty(out_features, buffer_size, device=self._device, dtype=self._dtype))
        
        self.reset_parameters(sigma_init)
        self.refresh_noise_stream()

    def reset_parameters(self, sigma_init):
        # ... (Same as before) ...
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / self.out_features ** 0.5)

    def refresh_noise_stream(self):
        # SELECT GENERATOR
        if self.method == 'cholesky':
            gen_func = generate_cholesky
        else:
            gen_func = generate_davies_harte
            
        self.noise_stream_in = gen_func(
            self.buffer_size, self.H, size=(self.in_features,), device=self.weight_mu.device, dtype=self._dtype, seed=self._seed
        )
        self.noise_stream_out = gen_func(
            self.buffer_size, self.H, size=(self.out_features,), device=self.weight_mu.device, dtype=self._dtype, seed=self._seed
        )
        self.step_counter = 0

    # ... (Forward and sample_noise same as before) ...
    def sample_noise(self):
        if self.step_counter >= self.buffer_size: self.refresh_noise_stream()
        eps_in = self.noise_stream_in[:, self.step_counter]
        eps_out = self.noise_stream_out[:, self.step_counter]
        def f(x): return x.sign().mul(x.abs().sqrt())
        self.weight_epsilon = f(eps_out).outer(f(eps_in))
        self.bias_epsilon = f(eps_out)
        self.step_counter += 1

    def forward(self, input):
        if self.training:
            self.sample_noise()
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)    
# ... (FBMNoisyLinear is above this in the file) ...

from .generators import fbm # Ensure this import is at the top

class FractionalPositionalEmbedding(nn.Module):
    """
    Positional Embedding using frozen fBm paths.
    Captures multi-scale fractal dependencies.
    """
    def __init__(self, max_len, d_model, H_range=(0.1, 0.9), method='davies_harte', device='cpu', dtype=torch.float32, seed: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.method = method
        self._device = torch.device(device)
        self._dtype = dtype
        self._seed = seed
        
        # 1. Create H values from H_range[0] to H_range[1]
        # We ensure they are within safe bounds [0.01, 0.99]
        h_min = max(0.01, H_range[0])
        h_max = min(0.99, H_range[1])
        Hs = torch.linspace(h_min, h_max, d_model, device=self._device, dtype=self._dtype)
        
        embeddings = []
        for i in range(d_model):
            # Generate path using selected method
            # We generate on CPU initially to save GPU memory for weights, 
            # then register as buffer moves it to device automatically.
            path = fbm(max_len, H=Hs[i].item(), size=(1,), method=self.method, device=self._device, dtype=self._dtype, seed=self._seed).squeeze()
            
            # Normalization (Critical for Embeddings to preserve gradient scale)
            path = (path - path.mean()) / (path.std() + 1e-6)
            embeddings.append(path)
            
        # Shape: (max_len, d_model)
        # We assume the fbm path length (max_len+1) needs to be trimmed or matched
        # fbm() returns n+1 points. We take the first max_len or last max_len? 
        # Usually exclude 0. Let's take 1 to max_len+1
        pe_tensor = torch.stack(embeddings, dim=1)
        
        # Crop if fbm generated n+1 and we want max_len
        pe_tensor = pe_tensor[:max_len, :]
        
        self.register_buffer('pe', pe_tensor.to(self._device, self._dtype))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Dim)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(0)}")
            
        return x + self.pe[:seq_len, :].unsqueeze(0)