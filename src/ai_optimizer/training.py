"""
Training orchestration for AI-powered Passivbot optimization.
"""

import torch
import numpy as np
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import time

from .gpu_agent import PPOAgent
from .environment import PassivbotEnvironment
from .utils import get_device, setup_gpu, log_gpu_memory


class AITrainer:
    """
    Main training class for AI-powered optimization.
    Orchestrates the training process and manages the learning loop.
    """
    
    def __init__(self,
                 config_path: str,
                 optimization_bounds: List[Tuple[float, float]],
                 save_dir: str = "ai_models",
                 log_dir: str = "ai_logs",
                 device: Optional[torch.device] = None):
        """
        Initialize the AI trainer.
        
        Args:
            config_path: Path to base Passivbot configuration
            optimization_bounds: Parameter bounds for optimization
            save_dir: Directory to save trained models
            log_dir: Directory to save training logs
            device: PyTorch device to use
        """
        self.config_path = config_path
        self.bounds = optimization_bounds
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device or get_device()
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup GPU
        self.gpu_config = setup_gpu()
        logging.info(f"GPU setup: {self.gpu_config}")
        
        # Initialize environment
        self.env = PassivbotEnvironment(
            config_path=config_path,
            optimization_bounds=optimization_bounds
        )
        
        # Calculate state and action dimensions
        sample_state = self.env.reset()
        self.state_dim = len(sample_state)
        self.action_dim = len(optimization_bounds)
        
        logging.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        
        # Initialize agent
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Training metrics
        self.training_history = {
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'best_performances': deque(maxlen=1000),
            'training_losses': deque(maxlen=1000),
            'convergence_rates': deque(maxlen=1000)
        }
        
        # Performance tracking
        self.best_global_performance = None
        self.best_global_config = None
        self.episodes_since_improvement = 0
        
    def train(self,
              num_episodes: int = 1000,
              max_episode_steps: int = 50,
              update_frequency: int = 10,
              save_frequency: int = 100,
              early_stopping_patience: int = 200,
              target_performance: Optional[float] = None) -> Dict[str, Any]:
        """
        Train the AI agent.
        
        Args:
            num_episodes: Total number of training episodes
            max_episode_steps: Maximum steps per episode
            update_frequency: How often to update the agent (in episodes)
            save_frequency: How often to save the model (in episodes)
            early_stopping_patience: Episodes without improvement before stopping
            target_performance: Target performance to achieve
            
        Returns:
            Training summary
        """
        logging.info(f"Starting AI training for {num_episodes} episodes")
        log_gpu_memory()
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Run episode
            episode_metrics = self._run_episode(max_episode_steps)
            
            # Update training history
            self._update_training_history(episode_metrics)
            
            # Periodic agent updates
            if (episode + 1) % update_frequency == 0:
                update_metrics = self.agent.update()
                if update_metrics:
                    self.training_history['training_losses'].append(
                        update_metrics.get('policy_loss', 0) + update_metrics.get('value_loss', 0)
                    )
            
            # Periodic saving
            if (episode + 1) % save_frequency == 0:
                self._save_checkpoint(episode, episode_metrics)
            
            # Logging
            episode_time = time.time() - episode_start_time
            self._log_episode_progress(episode, episode_metrics, episode_time)
            
            # Early stopping check
            if self._should_stop_early(early_stopping_patience, target_performance):
                logging.info(f"Early stopping at episode {episode}")
                break
            
            # GPU memory management
            if (episode + 1) % 50 == 0:
                log_gpu_memory()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # Final save
        final_metrics = self._save_final_model()
        
        # Training summary
        summary = self._get_training_summary(num_episodes, total_time)
        
        logging.info(f"Training completed in {total_time:.2f} seconds")
        return summary
    
    def _run_episode(self, max_steps: int) -> Dict[str, Any]:
        """Run a single training episode."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Get action from agent
            action, value, log_prob = self.agent.get_action_and_value(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            # Update for next iteration
            state = next_state
            total_reward += reward
            steps += 1
        
        # Get final value for GAE computation
        if not done:
            final_value = self.agent.get_value(state)
            self.agent.store_transition(state, action, 0, final_value, 0, True)
        
        # Get episode summary from environment
        env_summary = self.env.get_episode_summary()
        
        # Track best performance
        current_performance = info.get('performance', {})
        if self._is_new_best_performance(current_performance):
            self.best_global_performance = current_performance.copy()
            self.best_global_config = self.env.current_individual.copy()
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'done': done,
            'final_performance': current_performance,
            'env_summary': env_summary,
            'convergence_steps': steps if done else max_steps
        }
    
    def _update_training_history(self, episode_metrics: Dict[str, Any]):
        """Update training history with episode metrics."""
        self.training_history['episode_rewards'].append(episode_metrics['total_reward'])
        self.training_history['episode_lengths'].append(episode_metrics['steps'])
        
        if episode_metrics['final_performance']:
            perf_score = self._calculate_performance_score(episode_metrics['final_performance'])
            self.training_history['best_performances'].append(perf_score)
        
        self.training_history['convergence_rates'].append(
            episode_metrics['convergence_steps'] / episode_metrics['steps']
        )
    
    def _calculate_performance_score(self, performance: Dict[str, float]) -> float:
        """Calculate a single performance score from multiple metrics."""
        # Use the same weights as environment rewards
        weights = {
            'gain': 0.3,
            'sharpe_ratio': 0.2,
            'drawdown_worst': -0.2,
            'calmar_ratio': 0.15,
            'volume_pct_per_day_avg': 0.1,
            'position_held_hours_mean': -0.05
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in performance:
                score += weight * performance[metric]
        
        return score
    
    def _is_new_best_performance(self, performance: Dict[str, float]) -> bool:
        """Check if this is a new best performance."""
        if not performance or self.best_global_performance is None:
            return bool(performance)
        
        current_score = self._calculate_performance_score(performance)
        best_score = self._calculate_performance_score(self.best_global_performance)
        
        return current_score > best_score
    
    def _should_stop_early(self, patience: int, target_performance: Optional[float]) -> bool:
        """Check if training should stop early."""
        # Check patience
        if self.episodes_since_improvement >= patience:
            return True
        
        # Check target performance
        if target_performance is not None and self.best_global_performance:
            current_score = self._calculate_performance_score(self.best_global_performance)
            if current_score >= target_performance:
                return True
        
        return False
    
    def _save_checkpoint(self, episode: int, metrics: Dict[str, Any]):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}.pt")
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training state
        state_path = os.path.join(self.save_dir, f"training_state_{episode}.json")
        training_state = {
            'episode': episode,
            'best_global_performance': self.best_global_performance,
            'best_global_config': self.best_global_config.tolist() if self.best_global_config is not None else None,
            'episodes_since_improvement': self.episodes_since_improvement,
            'recent_metrics': metrics
        }
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logging.info(f"Checkpoint saved at episode {episode}")
    
    def _save_final_model(self) -> Dict[str, Any]:
        """Save final trained model."""
        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        self.agent.save(final_model_path)
        
        # Save best configuration
        if self.best_global_config is not None:
            best_config_path = os.path.join(self.save_dir, "best_config.json")
            best_config_data = {
                'parameters': self.best_global_config.tolist(),
                'performance': self.best_global_performance,
                'bounds': self.bounds,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(best_config_path, 'w') as f:
                json.dump(best_config_data, f, indent=2)
        
        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        serializable_history = {
            key: list(values) for key, values in self.training_history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        return {
            'model_path': final_model_path,
            'best_config_path': best_config_path if self.best_global_config is not None else None,
            'history_path': history_path
        }
    
    def _log_episode_progress(self, episode: int, metrics: Dict[str, Any], episode_time: float):
        """Log progress for current episode."""
        if episode % 10 == 0 or episode < 10:
            recent_rewards = list(self.training_history['episode_rewards'])[-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            performance = metrics.get('final_performance', {})
            gain = performance.get('gain', 0)
            sharpe = performance.get('sharpe_ratio', 0)
            
            logging.info(
                f"Episode {episode:4d} | "
                f"Reward: {metrics['total_reward']:7.4f} | "
                f"Avg Reward: {avg_reward:7.4f} | "
                f"Steps: {metrics['steps']:2d} | "
                f"Gain: {gain:6.3f} | "
                f"Sharpe: {sharpe:6.3f} | "
                f"Time: {episode_time:.2f}s"
            )
    
    def _get_training_summary(self, num_episodes: int, total_time: float) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'training_config': {
                'num_episodes': num_episodes,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'device': str(self.device),
                'total_time': total_time
            },
            'performance_metrics': {
                'best_global_performance': self.best_global_performance,
                'best_global_config': self.best_global_config.tolist() if self.best_global_config is not None else None,
                'episodes_since_improvement': self.episodes_since_improvement
            },
            'training_statistics': {
                'avg_episode_reward': np.mean(list(self.training_history['episode_rewards'])) if self.training_history['episode_rewards'] else 0,
                'max_episode_reward': np.max(list(self.training_history['episode_rewards'])) if self.training_history['episode_rewards'] else 0,
                'avg_episode_length': np.mean(list(self.training_history['episode_lengths'])) if self.training_history['episode_lengths'] else 0,
                'avg_convergence_rate': np.mean(list(self.training_history['convergence_rates'])) if self.training_history['convergence_rates'] else 0
            },
            'gpu_info': self.gpu_config
        }
        
        return summary
    
    def load_checkpoint(self, checkpoint_path: str, state_path: Optional[str] = None):
        """Load training checkpoint."""
        # Load agent
        self.agent.load(checkpoint_path)
        
        # Load training state if provided
        if state_path and os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            self.best_global_performance = training_state.get('best_global_performance')
            best_config = training_state.get('best_global_config')
            if best_config:
                self.best_global_config = np.array(best_config)
            self.episodes_since_improvement = training_state.get('episodes_since_improvement', 0)
        
        logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained agent."""
        logging.info(f"Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_performances = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 50:
                # Get action (no exploration)
                with torch.no_grad():
                    action, _, _ = self.agent.get_action_and_value(state)
                
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            if 'performance' in info:
                episode_performances.append(info['performance'])
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'avg_performance': {
                key: np.mean([p.get(key, 0) for p in episode_performances])
                for key in episode_performances[0].keys()
            } if episode_performances else {},
            'episode_rewards': episode_rewards,
            'episode_performances': episode_performances
        }