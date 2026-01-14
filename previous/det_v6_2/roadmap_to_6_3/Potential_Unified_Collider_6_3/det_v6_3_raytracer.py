import torch
import numpy as np
from typing import Tuple, Optional

class DETRaytracer:
    """
    DET v6.3 Raytracer - Volumetric Rendering and State Probing
    Uses ray-marching to visualize the DET lattice.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def render_volume(self, field: torch.Tensor, 
                      view_matrix: torch.Tensor, 
                      resolution: Tuple[int, int] = (128, 128),
                      num_steps: int = 64,
                      step_size: float = 0.5) -> torch.Tensor:
        """
        Render a 3D field using ray-marching.
        
        field: 3D tensor [N, N, N]
        view_matrix: 4x4 camera matrix
        resolution: (width, height) of output image
        """
        N = field.shape[0]
        W, H = resolution
        
        # 1. Generate rays in camera space
        x = torch.linspace(-1, 1, W, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        
        # Ray origins and directions
        ray_origins = torch.zeros((W, H, 3), device=self.device)
        ray_dirs = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        
        # 2. Transform rays to world space (simplified for now)
        # In a full implementation, we'd apply the view_matrix here
        
        # 3. Ray-marching
        accumulated_density = torch.zeros((W, H), device=self.device)
        current_pos = ray_origins + ray_dirs * (N / 4) # Start a bit back
        
        for _ in range(num_steps):
            # Sample field at current_pos
            # Map pos [-N/2, N/2] to [0, N-1]
            sample_pos = (current_pos + N/2).long()
            
            # Bounds check
            mask = (sample_pos[..., 0] >= 0) & (sample_pos[..., 0] < N) & \
                   (sample_pos[..., 1] >= 0) & (sample_pos[..., 1] < N) & \
                   (sample_pos[..., 2] >= 0) & (sample_pos[..., 2] < N)
            
            if torch.any(mask):
                # Vectorized sampling
                valid_pos = sample_pos[mask]
                densities = field[valid_pos[:, 0], valid_pos[:, 1], valid_pos[:, 2]]
                accumulated_density[mask] += densities * step_size
            
            current_pos += ray_dirs * step_size
            
        return accumulated_density

    def probe_ray(self, field: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor, 
                  max_dist: float, step_size: float = 0.1) -> torch.Tensor:
        """Probe the field along a single ray and return the profile."""
        steps = int(max_dist / step_size)
        profile = []
        N = field.shape[0]
        
        for i in range(steps):
            pos = origin + direction * (i * step_size)
            sample_pos = (pos + N/2).long()
            
            if (0 <= sample_pos[0] < N) and (0 <= sample_pos[1] < N) and (0 <= sample_pos[2] < N):
                profile.append(field[sample_pos[0], sample_pos[1], sample_pos[2]].item())
            else:
                profile.append(0.0)
                
        return torch.tensor(profile)

# Basic test
if __name__ == "__main__":
    tracer = DETRaytracer()
    field = torch.zeros((32, 32, 32))
    field[14:18, 14:18, 14:18] = 1.0 # A small cube in the center
    
    img = tracer.render_volume(field, torch.eye(4), resolution=(64, 64))
    print(f"Rendered image shape: {img.shape}")
    print(f"Max density in render: {torch.max(img).item():.4f}")
