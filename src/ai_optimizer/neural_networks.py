"""
Neural network architectures for the AI optimization agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for capturing complex parameter relationships."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Apply self-attention
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        # Residual connection and layer norm
        return self.layer_norm(output + x)


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs parameter modification actions.
    Uses attention mechanism to capture relationships between parameters.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Market condition encoder (LSTM for temporal patterns)
        self.market_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Self-attention layers for parameter relationships
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Parameter-specific processing
        self.param_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Action heads (mean and log_std for continuous actions)
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            action_mean: Mean of action distribution [batch_size, action_dim]
            action_std: Standard deviation of action distribution [batch_size, action_dim]
        """
        batch_size = state.size(0)
        
        # Input embedding
        x = F.relu(self.input_embedding(state))  # [batch_size, hidden_dim]
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Market pattern encoding
        lstm_out, _ = self.market_encoder(x)  # [batch_size, 1, hidden_dim]
        
        # Self-attention for parameter relationships
        for attention_layer in self.attention_layers:
            lstm_out = attention_layer(lstm_out)
        
        # Remove sequence dimension
        x = lstm_out.squeeze(1)  # [batch_size, hidden_dim]
        
        # Parameter processing
        x = self.param_processor(x)  # [batch_size, hidden_dim // 2]
        
        # Action distribution parameters
        action_mean = self.action_mean(x)  # [batch_size, action_dim]
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        
        return action_mean, action_std


class ValueNetwork(nn.Module):
    """
    Value network that estimates expected returns.
    Shares some architecture with policy network for efficiency.
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Market condition encoder
        self.market_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            value: Estimated value [batch_size, 1]
        """
        # Input embedding
        x = F.relu(self.input_embedding(state))
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)
        
        # Market pattern encoding
        lstm_out, _ = self.market_encoder(x)
        
        # Self-attention
        for attention_layer in self.attention_layers:
            lstm_out = attention_layer(lstm_out)
            
        # Remove sequence dimension
        x = lstm_out.squeeze(1)
        
        # Value estimation
        value = self.value_head(x)
        
        return value


class SharedBackbone(nn.Module):
    """
    Shared backbone for both policy and value networks.
    Reduces memory usage and training time.
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Market encoder
        self.market_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Extract shared features from state."""
        # Input embedding
        x = F.relu(self.input_embedding(state))
        
        # Add sequence dimension
        x = x.unsqueeze(1)
        
        # Market encoding and attention
        lstm_out, _ = self.market_encoder(x)
        
        for attention_layer in self.attention_layers:
            lstm_out = attention_layer(lstm_out)
            
        # Remove sequence dimension
        return lstm_out.squeeze(1)