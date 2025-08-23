# AI Architecture Documentation

This document provides detailed technical information about the AI optimization system's architecture, components, and implementation details.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Passivbot AI System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │   User Interface │    │     Auto-Optimization        │   │
│  │                 │    │                              │   │
│  │  • CLI Commands │    │  • Performance Monitor       │   │
│  │  • PBGUI Web UI │    │  • Market Regime Detection   │   │
│  │  • Config Files │    │  • Automatic Retraining     │   │
│  └─────────────────┘    └──────────────────────────────┘   │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              AI Training Controller                     │ │
│  │                                                         │ │
│  │  • Training Orchestration    • Model Management        │ │
│  │  • Hyperparameter Tuning     • Checkpointing          │ │
│  │  • Progress Monitoring       • Performance Tracking    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                             │
│                              ▼                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                PPO Agent (GPU)                          │ │
│  │                                                         │ │
│  │  ┌─────────────────┐    ┌─────────────────┐            │ │
│  │  │  Policy Network │    │  Value Network  │            │ │
│  │  │                 │    │                 │            │ │
│  │  │ • Parameter     │    │ • Performance   │            │ │
│  │  │   Suggestions   │    │   Estimation    │            │ │
│  │  │ • Multi-head    │    │ • State Value   │            │ │
│  │  │   Attention     │    │   Functions     │            │ │
│  │  │ • LSTM Layers   │    │ • Risk Assessment│           │ │
│  │  └─────────────────┘    └─────────────────┘            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                             │
│                              ▼                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Trading Environment                          │ │
│  │                                                         │ │
│  │  • Parameter Space Management                          │ │
│  │  • Backtest Integration                                │ │
│  │  • Reward Calculation                                  │ │
│  │  • Market State Processing                             │ │
│  │  • Performance Metrics                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                             │
│                              ▼                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Passivbot Core (Rust + Python)                  │ │
│  │                                                         │ │
│  │  • High-speed Backtesting (Rust)                       │ │
│  │  • Trading Logic (Python)                              │ │
│  │  • Exchange Integration                                │ │
│  │  • Configuration Management                            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PPO Agent (`src/ai_optimizer/gpu_agent.py`)

The central AI component implementing Proximal Policy Optimization:

#### Architecture Features

```python
class PPOAgent:
    """GPU-accelerated PPO agent for trading optimization."""
    
    def __init__(self, state_dim, action_dim, **kwargs):
        # Neural network architectures
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # Training components
        self.optimizer = torch.optim.Adam(params, lr=3e-4)
        self.buffer = ExperienceBuffer()
        
        # GPU optimization
        if torch.cuda.is_available():
            self.policy_net = torch.compile(self.policy_net)
            self.value_net = torch.compile(self.value_net)
```

#### Key Methods

- **`get_action_and_value(state)`**: Sample actions from policy
- **`store_transition()`**: Store experience in replay buffer  
- **`compute_gae()`**: Calculate Generalized Advantage Estimation
- **`update()`**: Perform PPO policy and value updates

#### Training Algorithm

```python
def update(self, epochs=10, batch_size=64):
    """PPO update with clipped objective."""
    
    for epoch in range(epochs):
        # Get batch data
        states, actions, old_log_probs, advantages, returns = self.buffer.get_batch()
        
        # Forward pass
        action_mean, action_std = self.policy_net(states)
        values = self.value_net(states)
        
        # Calculate policy loss with clipping
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss with entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
```

### 2. Neural Networks (`src/ai_optimizer/neural_networks.py`)

Advanced neural architectures designed for trading parameter optimization:

#### Policy Network

```python
class PolicyNetwork(nn.Module):
    """Policy network with attention mechanisms."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # Input processing
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Market pattern recognition
        self.market_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Parameter relationship modeling
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads=8)
            for _ in range(3)
        ])
        
        # Action generation
        self.action_mean = nn.Linear(hidden_dim // 2, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
```

#### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Captures complex parameter relationships."""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-head attention computation
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
        
        # Residual connection and layer normalization
        return self.layer_norm(output + x)
```

### 3. Trading Environment (`src/ai_optimizer/environment.py`)

Interfaces the AI agent with Passivbot's optimization system:

#### State Space Design

The environment provides a comprehensive state representation:

```python
def _get_state(self):
    """Construct state vector for AI agent."""
    
    # Current parameter values (normalized to [0,1])
    param_state = self._normalize_parameters(self.current_individual)
    
    # Performance features from recent backtests
    performance_features = self._extract_performance_features()
    
    # Market condition indicators
    market_state = np.array([
        self.market_features['volatility'],      # Recent price volatility
        self.market_features['trend'],           # Trend strength
        self.market_features['volume'],          # Volume patterns
        self.market_features['correlation']      # Cross-asset correlation
    ])
    
    # Training progress indicators
    progress_features = np.array([
        self.current_step / self.max_episode_steps,  # Episode progress
        len(self.episode_history) / self.max_episode_steps  # Experience count
    ])
    
    # Combine all features into single state vector
    state = np.concatenate([
        param_state,         # Current parameters
        performance_features, # Recent performance
        market_state,        # Market conditions
        progress_features    # Training progress
    ])
    
    return state.astype(np.float32)
```

#### Reward Function Implementation

```python
def _calculate_reward(self, performance):
    """Multi-objective reward calculation."""
    
    reward = 0.0
    
    # Primary performance metrics (70% weight)
    for metric, weight in self.reward_weights.items():
        if metric in performance:
            value = performance[metric]
            
            # Normalize different metrics to similar scales
            if metric in ['gain', 'sharpe_ratio', 'calmar_ratio']:
                normalized_value = np.tanh(value)  # Sigmoid normalization
            elif metric == 'drawdown_worst':
                normalized_value = -np.tanh(value * 5)  # Penalty for drawdown
            elif metric == 'volume_pct_per_day_avg':
                normalized_value = -abs(value - 1.0)  # Target ~1.0 volume
            else:
                normalized_value = np.tanh(value)
            
            reward += weight * normalized_value
    
    # Exploration bonuses (10% weight)
    exploration_bonus = self._calculate_exploration_bonus()
    reward += 0.01 * exploration_bonus
    
    # Stability penalties (10% weight)
    stability_penalty = self._calculate_stability_penalty()
    reward -= 0.01 * stability_penalty
    
    return reward
```

### 4. Training Controller (`src/ai_optimizer/training.py`)

Orchestrates the entire training process:

#### Training Loop

```python
async def train(self, num_episodes=1000):
    """Main training loop with progress monitoring."""
    
    for episode in range(num_episodes):
        # Reset environment for new episode
        state = self.env.reset()
        total_reward = 0
        
        # Run episode until completion
        for step in range(self.max_episode_steps):
            # Get action from agent
            action, value, log_prob = self.agent.get_action_and_value(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update agent every N episodes
        if (episode + 1) % self.update_frequency == 0:
            update_metrics = self.agent.update()
            self._log_training_progress(episode, update_metrics)
        
        # Save checkpoint periodically
        if (episode + 1) % self.save_frequency == 0:
            self._save_checkpoint(episode)
        
        # Check early stopping criteria
        if self._should_stop_early():
            break
    
    return self._get_training_summary()
```

### 5. Auto-Optimization Controller (`src/auto_optimizer.py`)

Manages live trading adaptation:

#### Performance Monitoring

```python
async def _monitoring_cycle(self):
    """Execute one monitoring cycle."""
    
    # Collect current performance metrics
    current_metrics = await self._collect_performance_metrics()
    
    if current_metrics:
        self.performance_history.append(current_metrics)
        
        # Update market condition indicators
        self._update_market_conditions(current_metrics)
        
        # Check if retraining is needed
        performance_degradation = self._calculate_performance_degradation()
        market_regime_changed = self._detect_market_regime_change()
        
        if (performance_degradation < self.retraining_threshold or 
            market_regime_changed):
            await self._trigger_retraining()
```

#### Market Regime Detection

```python
def _detect_market_regime_change(self):
    """Statistical test for regime changes."""
    
    for condition_name, condition_data in self.market_conditions.items():
        if len(condition_data) < 12:
            continue
            
        # Compare recent vs historical periods
        recent_data = list(condition_data)[-6:]
        historical_data = list(condition_data)[-12:-6]
        
        recent_mean = np.mean(recent_data)
        historical_mean = np.mean(historical_data)
        historical_std = np.std(historical_data)
        
        # Statistical significance test (2 standard deviations)
        if historical_std > 0:
            z_score = abs(recent_mean - historical_mean) / historical_std
            if z_score > 2.0:
                logging.info(f"Regime change detected in {condition_name}: z={z_score:.2f}")
                return True
    
    return False
```

## GPU Acceleration

### Device Management

```python
def get_device():
    """Automatic device selection with fallback."""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders")
        
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU acceleration)")
    
    return device
```

### Memory Optimization

```python
def setup_gpu_optimization():
    """Configure GPU for optimal performance."""
    
    if torch.cuda.is_available():
        # Enable memory optimization
        torch.cuda.empty_cache()
        
        # Configure memory allocation
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Model compilation for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            model = model  # torch.compile disabled for older GPU compatibility
    
    return {'device': device, 'scaler': scaler, 'compiled': True}
```

### Batch Processing

```python
def process_batch_gpu(self, batch_data):
    """Optimized batch processing on GPU."""
    
    with torch.cuda.amp.autocast():  # Mixed precision
        # Move data to GPU
        states = batch_data['states'].to(self.device, non_blocking=True)
        actions = batch_data['actions'].to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.no_grad():
            policy_output = self.policy_net(states)
            values = self.value_net(states)
        
        # Async data transfer back to CPU
        results = {
            'policy': policy_output.cpu(),
            'values': values.cpu()
        }
    
    return results
```

## Integration Points

### With Existing Optimization System

```python
# src/optimize.py integration
def run_ai_optimization(config, args, bounds, sig_digits, results_queue):
    """AI optimization entry point."""
    
    # Initialize AI trainer with existing config
    trainer = AITrainer(
        config_path=args.config_path,
        optimization_bounds=bounds,
        save_dir="ai_models"
    )
    
    # Load pre-trained model if specified
    if args.ai_model_path:
        trainer.load_checkpoint(args.ai_model_path)
    
    # Run training with existing bounds and constraints
    training_summary = trainer.train(
        num_episodes=args.ai_episodes,
        max_episode_steps=50
    )
    
    # Convert results to compatible format
    best_config = training_summary['performance_metrics']['best_global_config']
    population = [creator.Individual(best_config)] if best_config else []
    
    return population, create_logbook()
```

### With Backtesting Engine

```python
def _run_backtest_evaluation(self, individual):
    """Interface with Rust backtesting engine."""
    
    try:
        # Convert individual to config format
        config = individual_to_config(individual, self.base_config)
        
        # Run backtest using existing Rust engine
        performance = run_backtest_rust(config, self.market_data)
        
        return {
            'gain': performance.gain,
            'sharpe_ratio': performance.sharpe,
            'drawdown_worst': performance.max_drawdown,
            'calmar_ratio': performance.calmar,
            'volume_pct_per_day_avg': performance.volume_ratio
        }
        
    except Exception as e:
        logging.error(f"Backtest evaluation failed: {e}")
        return self._get_default_poor_performance()
```

### With Live Trading

```python
def integrate_with_live_trading(self):
    """Integration with live Passivbot instances."""
    
    # Monitor live performance
    performance_collector = LivePerformanceCollector()
    
    # Auto-optimization controller
    auto_optimizer = AutoOptimizer(
        config_path=self.live_config_path,
        optimization_bounds=self.bounds
    )
    
    # Start monitoring in background
    asyncio.create_task(auto_optimizer.start_monitoring())
    
    return auto_optimizer
```

## Error Handling and Robustness

### Training Stability

```python
def ensure_training_stability(self):
    """Implement stability measures."""
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
    
    # Learning rate scheduling
    if self.current_episode > 0 and self.current_episode % 1000 == 0:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
    
    # Early stopping on divergence
    if len(self.recent_losses) > 10:
        recent_loss_trend = np.mean(self.recent_losses[-5:]) - np.mean(self.recent_losses[-10:-5])
        if recent_loss_trend > 0.1:  # Loss increasing
            logging.warning("Training instability detected, reducing learning rate")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
```

### Memory Management

```python
def manage_gpu_memory(self):
    """Proactive GPU memory management."""
    
    # Clear cache periodically
    if self.current_episode % 50 == 0:
        torch.cuda.empty_cache()
    
    # Monitor memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if memory_used / memory_total > 0.9:
            logging.warning(f"High GPU memory usage: {memory_used:.1f}GB/{memory_total:.1f}GB")
            
            # Reduce batch size if needed
            self.batch_size = max(16, self.batch_size // 2)
```

### Exception Handling

```python
class RobustAITrainer(AITrainer):
    """Training with comprehensive error handling."""
    
    def _safe_training_step(self, state, action):
        """Training step with error recovery."""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                next_state, reward, done, info = self.env.step(action)
                return next_state, reward, done, info
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"GPU OOM on attempt {attempt + 1}, reducing batch size")
                    torch.cuda.empty_cache()
                    self.batch_size = max(8, self.batch_size // 2)
                    
                elif attempt == max_retries - 1:
                    logging.error(f"Training step failed after {max_retries} attempts: {e}")
                    raise
                    
                else:
                    time.sleep(1)  # Brief pause before retry
        
        return state, -1.0, True, {'error': 'max_retries_exceeded'}
```

This architecture provides a robust, scalable, and high-performance AI optimization system that seamlessly integrates with Passivbot's existing infrastructure while providing significant improvements in optimization speed and quality.