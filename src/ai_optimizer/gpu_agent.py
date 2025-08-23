"""
GPU-accelerated PPO agent for Passivbot optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from .neural_networks import PolicyNetwork, ValueNetwork, SharedBackbone
from .utils import get_device, clip_gradients, tensor_to_numpy, numpy_to_tensor


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading strategy optimization.
    Designed specifically for Passivbot parameter optimization.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 shared_backbone: bool = True,
                 device: Optional[torch.device] = None):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            shared_backbone: Whether to use shared backbone for policy and value
            device: PyTorch device
        """
        self.device = device or get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize networks
        if shared_backbone:
            self.backbone = SharedBackbone(state_dim).to(self.device)
            self.policy_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim * 2)  # mean and log_std
            ).to(self.device)
            self.value_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(self.device)
            
            # Combined optimizer
            all_params = list(self.backbone.parameters()) + \
                        list(self.policy_head.parameters()) + \
                        list(self.value_head.parameters())
            self.optimizer = torch.optim.Adam(all_params, lr=lr)
            
        else:
            self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
            self.value_net = ValueNetwork(state_dim).to(self.device)
            
            # Separate optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.shared_backbone = shared_backbone
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
        # Enable model compilation for additional speedup on modern GPUs
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                if self.shared_backbone:
                    self.backbone = self.backbone
                else:
                    self.policy_net = self.policy_net
                    self.value_net = self.value_net
                logging.info("Models compiled for additional GPU acceleration")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
    
    def get_action_and_value(self, state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Get action, value, and log probability for given state.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            value: State value estimate
            log_prob: Log probability of action
        """
        with torch.no_grad():
            state_tensor = numpy_to_tensor(state.reshape(1, -1), self.device)
            
            if self.shared_backbone:
                features = self.backbone(state_tensor)
                
                # Policy output
                policy_out = self.policy_head(features)
                action_mean = policy_out[:, :self.action_dim]
                action_log_std = policy_out[:, self.action_dim:]
                action_std = torch.exp(action_log_std)
                
                # Value output
                value = self.value_head(features)
                
            else:
                action_mean, action_std = self.policy_net(state_tensor)
                value = self.value_net(state_tensor)
            
            # Sample action from normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Convert to numpy
            action_np = tensor_to_numpy(action[0])
            value_np = tensor_to_numpy(value[0, 0])
            log_prob_np = tensor_to_numpy(log_prob[0])
            
        return action_np, value_np, log_prob_np
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            state_tensor = numpy_to_tensor(state.reshape(1, -1), self.device)
            
            if self.shared_backbone:
                features = self.backbone(state_tensor)
                value = self.value_head(features)
            else:
                value = self.value_net(state_tensor)
            
            return tensor_to_numpy(value[0, 0])
    
    def store_transition(self, 
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        value: float,
                        log_prob: float,
                        done: bool):
        """Store transition in experience buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[t])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, 
               epochs: int = 10,
               batch_size: int = 64,
               early_stopping_kl: float = 0.01) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            epochs: Number of update epochs
            batch_size: Batch size for updates
            early_stopping_kl: KL divergence threshold for early stopping
            
        Returns:
            Training statistics
        """
        if len(self.buffer) < batch_size:
            return {}
        
        # Get batch data
        batch = self.buffer.get_batch()
        states = numpy_to_tensor(batch['states'], self.device)
        actions = numpy_to_tensor(batch['actions'], self.device)
        old_log_probs = numpy_to_tensor(batch['log_probs'], self.device)
        advantages, returns = self.compute_gae(
            batch['rewards'], batch['values'], batch['dones']
        )
        advantages = numpy_to_tensor(advantages, self.device)
        returns = numpy_to_tensor(returns, self.device)
        
        # Training loop
        epoch_stats = []
        
        for epoch in range(epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                if self.shared_backbone:
                    features = self.backbone(batch_states)
                    
                    # Policy
                    policy_out = self.policy_head(features)
                    action_mean = policy_out[:, :self.action_dim]
                    action_log_std = policy_out[:, self.action_dim:]
                    action_std = torch.exp(action_log_std)
                    
                    # Value
                    values = self.value_head(features).squeeze()
                    
                else:
                    action_mean, action_std = self.policy_net(batch_states)
                    values = self.value_net(batch_states).squeeze()
                
                # Compute policy loss
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                if self.shared_backbone:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    clip_gradients(self.backbone, self.max_grad_norm)
                    clip_gradients(self.policy_head, self.max_grad_norm)
                    clip_gradients(self.value_head, self.max_grad_norm)
                    self.optimizer.step()
                else:
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    clip_gradients(self.policy_net, self.max_grad_norm)
                    clip_gradients(self.value_net, self.max_grad_norm)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                # Compute KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - log_probs).mean()
                
                # Store statistics
                epoch_stats.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'kl_divergence': kl_div.item()
                })
                
                # Early stopping based on KL divergence
                if kl_div > early_stopping_kl:
                    logging.info(f"Early stopping at epoch {epoch} due to KL divergence: {kl_div:.4f}")
                    break
            
            if len(epoch_stats) > 0 and epoch_stats[-1]['kl_divergence'] > early_stopping_kl:
                break
        
        # Update training statistics
        if epoch_stats:
            avg_stats = {
                key: np.mean([stat[key] for stat in epoch_stats])
                for key in epoch_stats[0].keys()
            }
            
            for key, value in avg_stats.items():
                self.training_stats[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return avg_stats if epoch_stats else {}
    
    def save(self, filepath: str):
        """Save agent to file."""
        if self.shared_backbone:
            torch.save({
                'backbone': self.backbone.state_dict(),
                'policy_head': self.policy_head.state_dict(),
                'value_head': self.value_head.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_stats': dict(self.training_stats)
            }, filepath)
        else:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'value_net': self.value_net.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
                'training_stats': dict(self.training_stats)
            }, filepath)
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.shared_backbone:
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.policy_head.load_state_dict(checkpoint['policy_head'])
            self.value_head.load_state_dict(checkpoint['value_head'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


class ExperienceBuffer:
    """Buffer for storing agent experiences."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get_batch(self):
        """Get all stored transitions as a batch."""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones
        }
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)