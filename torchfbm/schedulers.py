def get_hurst_schedule(n_steps: int, start_H: float = 0.3, end_H: float = 0.7, type='linear'):
    """
    Returns a schedule of H values for Diffusion sampling.
    """
    if type == 'linear':
        return torch.linspace(start_H, end_H, n_steps)
    elif type == 'cosine':
        # Cosine annealing
        steps = torch.arange(n_steps)
        return end_H + 0.5 * (start_H - end_H) * (1 + torch.cos(steps / n_steps * torch.pi))