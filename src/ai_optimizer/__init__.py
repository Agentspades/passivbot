"""
AI-powered optimization module for Passivbot.

This module implements GPU-accelerated reinforcement learning
for automatic trading strategy optimization.
"""

from .gpu_agent import PPOAgent
from .environment import PassivbotEnvironment
from .neural_networks import PolicyNetwork, ValueNetwork
from .training import AITrainer
from .utils import get_device, setup_gpu

__all__ = [
    "PPOAgent",
    "PassivbotEnvironment", 
    "PolicyNetwork",
    "ValueNetwork",
    "AITrainer",
    "get_device",
    "setup_gpu"
]