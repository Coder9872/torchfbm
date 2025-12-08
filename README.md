# torch-fbm

Fractional Brownian motion (fBm) and fractional Gaussian noise (fGn) generators, layers, and processes in PyTorch.

Features
- Fast Davies–Harte and exact Cholesky fGn generators: see `torch-fbm/generators.generate_davies_harte` and `torch-fbm/generators.generate_cholesky`.
- fBm paths via cumulative sum: `torch-fbm/generators.fbm`.
- Fractional OU and Geometric fBm processes: `torch-fbm/processes.fractional_ou_process`, `torch-fbm/processes.geometric_fbm`.
- Noisy linear layer and positional embeddings using fBm: `torch-fbm/layers.FBMNoisyLinear`, `torch-fbm/layers.FractionalPositionalEmbedding`.
- Hurst exponent estimator: `torch-fbm/estimators.estimate_hurst`.
- RL action noise stream: `torch-fbm/rl.FBMActionNoise`.

Install
- Editable install:
	- In VS Code terminal:
		- `pip install -e .`

Quick Usage
- Generate fBm (torch-only):
	- Python:
		- `import torch`
		- `from torch_fbm import fbm, get_default_device, set_seed`
		- `device = get_default_device(); set_seed(42)`
		- `path = fbm(n=1024, H=0.7, size=(4,), method='davies_harte', device=device, dtype=torch.float32)`
- Noisy Linear:
	- Python:
		- `from torch_fbm import FBMNoisyLinear`
		- `layer = FBMNoisyLinear(32, 10, H=0.5, method='davies_harte', device=device, dtype=torch.float32, seed=123)`
		- `x = torch.randn(8, 32, device=device)`
		- `layer.train(); y1 = layer(x); y2 = layer(x)`
		- `layer.eval(); y3 = layer(x)`
- Processes:
	- Python:
		- `from torch_fbm import geometric_fbm, fractional_ou_process`
		- `s = geometric_fbm(n=1000, H=0.7, mu=0.05, sigma=0.2, t_max=1.0, s0=100.0, device=device, dtype=torch.float32)`
		- `x = fractional_ou_process(n=2048, H=0.6, theta=0.2, mu=0.0, sigma=0.5, dt=1/256, device=device, dtype=torch.float32)`

Hurst Estimation
- `from torch_fbm import estimate_hurst`
- `H = estimate_hurst(path.unsqueeze(0), min_lag=4, max_lag=64, assume_path=True)`

RL Action Noise
- `from torch_fbm import FBMActionNoise`
- `noise = FBMActionNoise(mean=0.0, sigma=0.2, H=0.7, size=(1,), buffer_size=10000, method='davies_harte', device=device, return_torch=True, dtype=torch.float32, seed=7)`
- `a = noise()`

Notes
- Choose `method='davies_harte'` for speed ($O(N \log N)$), `cholesky` for validation ($O(N^3)$).
- `H` is clamped to $[0.01, 0.99]$ for stability.
- Set `dtype` and `seed` in generators for reproducibility.
- Torch-only: examples avoid `.numpy()` to work even if NumPy isn't available.
