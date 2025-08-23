"""
Utility functions for GPU acceleration and AI optimization.
"""

import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_device() -> torch.device:
    """Get the best available device (CUDA GPU, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def setup_gpu() -> Dict[str, Any]:
    """Setup GPU environment and return configuration."""
    device = get_device()
    
    config = {
        "device": device,
        "device_type": device.type,
        "mixed_precision": device.type == "cuda",
        "compile_model": device.type == "cuda" and hasattr(torch, "compile")
    }
    
    if device.type == "cuda":
        # Enable optimizations for CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory management
        torch.cuda.empty_cache()
        
        config.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return config


def normalize_state(state: np.ndarray, 
                   state_mean: Optional[np.ndarray] = None,
                   state_std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize state features for better RL training."""
    if state_mean is None:
        state_mean = np.mean(state, axis=0)
    if state_std is None:
        state_std = np.std(state, axis=0) + 1e-8
    
    normalized_state = (state - state_mean) / state_std
    return normalized_state, state_mean, state_std


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array safely."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def numpy_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor on specified device."""
    return torch.from_numpy(array).float().to(device)


def log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_model_checkpoint(model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         filepath: str):
    """Save model checkpoint with optimizer state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         filepath: str,
                         device: torch.device) -> Tuple[int, float]:
    """Load model checkpoint and return epoch and loss."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']