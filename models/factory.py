# models/factory.py
import torch, torch.nn as nn

class IdentityEnhancer(nn.Module):
    """Passthrough simplu (bun pentru test flux)."""
    def __init__(self): super().__init__()
    def forward(self, x):
        # acceptă [B,T] sau [B,1,T]; întoarce [B,T]
        if x.dim()==3: x = x[:,0,:]
        return x

def build_model_from_cfg(cfg):
    # Poți comuta ușor pe arhitectura ta reală aici.
    return IdentityEnhancer()
