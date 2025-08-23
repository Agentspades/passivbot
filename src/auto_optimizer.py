"""
Auto-optimization controller for live trading adaptation.
Monitors performance and triggers AI retraining when needed.
"""

import asyncio
import logging
import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
import threading

# Import AI modules if available
try:
    from ai_optimizer import AITrainer, setup_gpu
    AI_OPTIMIZATION_AVAILABLE = True
except ImportError:
    AI_OPTIMIZATION_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    balance: float
    gain: float
    drawdown: float
    sharpe_ratio: float
    volume: float
    num_trades: int
    win_rate: float


class AutoOptimizer:
    """
    Automatic optimization controller that monitors live trading performance
    and triggers AI retraining when performance degrades.
    """
    
    def __init__(self,
                 config_path: str,
                 optimization_bounds: List[tuple],
                 monitoring_interval: int = 3600,  # 1 hour
                 retraining_threshold: float = -0.1,  # 10% performance drop
                 min_retraining_interval: int = 86400,  # 24 hours
                 performance_window: int = 168,  # 7 days in hours
                 save_dir: str = "auto_optimizer"):
        """
        Initialize auto-optimizer.
        
        Args:
            config_path: Path to base configuration
            optimization_bounds: Parameter bounds for optimization
            monitoring_interval: How often to check performance (seconds)
            retraining_threshold: Performance drop threshold to trigger retraining
            min_retraining_interval: Minimum time between retrainings (seconds)
            performance_window: Window for performance comparison (hours)
            save_dir: Directory to save models and logs
        """
        self.config_path = config_path
        self.optimization_bounds = optimization_bounds
        self.monitoring_interval = monitoring_interval
        self.retraining_threshold = retraining_threshold
        self.min_retraining_interval = min_retraining_interval
        self.performance_window = performance_window
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=performance_window * 2)
        self.baseline_performance: Optional[PerformanceMetrics] = None
        self.last_retraining_time: Optional[datetime] = None
        self.current_model_path: Optional[str] = None
        
        # Auto-optimization state
        self.is_running = False
        self.is_retraining = False
        self.retraining_thread: Optional[threading.Thread] = None
        
        # Market condition detection
        self.market_conditions = {
            'volatility': deque(maxlen=24),  # 24 hours of volatility data
            'trend': deque(maxlen=24),
            'volume': deque(maxlen=24)
        }
        
        # Configuration adaptation
        self.adaptive_parameters = {}
        self.parameter_performance_map = {}
        
        logging.info(f"AutoOptimizer initialized with {len(optimization_bounds)} parameters")
        
        if not AI_OPTIMIZATION_AVAILABLE:
            logging.warning("AI optimization not available. Auto-optimizer will work in monitoring-only mode.")
    
    async def start_monitoring(self):
        """Start the monitoring loop."""
        if self.is_running:
            logging.warning("Auto-optimizer is already running")
            return
        
        self.is_running = True
        logging.info("Starting auto-optimization monitoring...")
        
        try:
            while self.is_running:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
        finally:
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.is_running = False
        if self.retraining_thread and self.retraining_thread.is_alive():
            logging.info("Waiting for retraining to complete...")
            self.retraining_thread.join(timeout=300)  # 5 minutes timeout
    
    async def _monitoring_cycle(self):
        """Execute one monitoring cycle."""
        try:
            # Collect current performance metrics
            current_metrics = await self._collect_performance_metrics()
            
            if current_metrics:
                self.performance_history.append(current_metrics)
                
                # Update market conditions
                self._update_market_conditions(current_metrics)
                
                # Check if retraining is needed
                if self._should_retrain(current_metrics):
                    await self._trigger_retraining()
                
                # Adaptive parameter adjustment (real-time micro-adjustments)
                await self._adjust_parameters_if_needed(current_metrics)
                
                # Log monitoring status
                self._log_monitoring_status(current_metrics)
        
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}")
    
    async def _collect_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Collect current performance metrics from the trading bot.
        This should interface with the actual Passivbot instance.
        """
        # TODO: Interface with actual Passivbot live trading data
        # For now, return simulated metrics
        
        try:
            # In a real implementation, this would:
            # 1. Connect to the live trading bot
            # 2. Fetch current balance, positions, trades
            # 3. Calculate performance metrics
            
            # Simulated metrics for demonstration
            now = datetime.now()
            
            # Generate realistic simulated performance
            if len(self.performance_history) > 0:
                last_balance = self.performance_history[-1].balance
                balance_change = np.random.normal(0.001, 0.01)  # Small random change
                new_balance = last_balance * (1 + balance_change)
            else:
                new_balance = 10000.0  # Starting balance
            
            metrics = PerformanceMetrics(
                timestamp=now,
                balance=new_balance,
                gain=np.random.normal(0.02, 0.1),
                drawdown=np.random.uniform(0.0, 0.05),
                sharpe_ratio=np.random.normal(1.5, 0.5),
                volume=np.random.uniform(100, 1000),
                num_trades=np.random.randint(5, 20),
                win_rate=np.random.uniform(0.4, 0.7)
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {e}")
            return None
    
    def _update_market_conditions(self, metrics: PerformanceMetrics):
        """Update market condition indicators."""
        # Calculate volatility based on balance changes
        if len(self.performance_history) >= 2:
            recent_balances = [m.balance for m in list(self.performance_history)[-10:]]
            volatility = np.std(recent_balances) / np.mean(recent_balances)
            self.market_conditions['volatility'].append(volatility)
        
        # Calculate trend
        if len(self.performance_history) >= 5:
            recent_gains = [m.gain for m in list(self.performance_history)[-5:]]
            trend = np.mean(recent_gains)
            self.market_conditions['trend'].append(trend)
        
        # Volume indicator
        self.market_conditions['volume'].append(metrics.volume)
    
    def _should_retrain(self, current_metrics: PerformanceMetrics) -> bool:
        """Determine if retraining should be triggered."""
        if not AI_OPTIMIZATION_AVAILABLE or self.is_retraining:
            return False
        
        # Check minimum interval since last retraining
        if (self.last_retraining_time and 
            datetime.now() - self.last_retraining_time < timedelta(seconds=self.min_retraining_interval)):
            return False
        
        # Need enough historical data for comparison
        if len(self.performance_history) < 24:  # At least 24 data points
            return False
        
        # Calculate performance degradation
        performance_degradation = self._calculate_performance_degradation(current_metrics)
        
        # Check market regime change
        market_regime_changed = self._detect_market_regime_change()
        
        # Trigger retraining if performance dropped significantly or market regime changed
        should_retrain = (performance_degradation < self.retraining_threshold or 
                         market_regime_changed)
        
        if should_retrain:
            logging.info(f"Retraining triggered - Performance degradation: {performance_degradation:.3f}, "
                        f"Market regime changed: {market_regime_changed}")
        
        return should_retrain
    
    def _calculate_performance_degradation(self, current_metrics: PerformanceMetrics) -> float:
        """Calculate performance degradation compared to baseline."""
        if len(self.performance_history) < 24:
            return 0.0
        
        # Use recent performance vs baseline
        recent_metrics = list(self.performance_history)[-12:]  # Last 12 periods
        baseline_metrics = list(self.performance_history)[-24:-12]  # Previous 12 periods
        
        recent_performance = np.mean([m.gain for m in recent_metrics])
        baseline_performance = np.mean([m.gain for m in baseline_metrics])
        
        if baseline_performance == 0:
            return 0.0
        
        degradation = (recent_performance - baseline_performance) / abs(baseline_performance)
        return degradation
    
    def _detect_market_regime_change(self) -> bool:
        """Detect if market regime has changed significantly."""
        if not all(len(cond) >= 12 for cond in self.market_conditions.values()):
            return False
        
        # Compare recent vs historical market conditions
        for condition_name, condition_data in self.market_conditions.items():
            recent_data = list(condition_data)[-6:]
            historical_data = list(condition_data)[-12:-6]
            
            recent_mean = np.mean(recent_data)
            historical_mean = np.mean(historical_data)
            historical_std = np.std(historical_data)
            
            if historical_std > 0:
                z_score = abs(recent_mean - historical_mean) / historical_std
                if z_score > 2.0:  # Significant change (2 standard deviations)
                    logging.info(f"Market regime change detected in {condition_name}: z-score = {z_score:.2f}")
                    return True
        
        return False
    
    async def _trigger_retraining(self):
        """Trigger AI model retraining in a separate thread."""
        if self.is_retraining:
            logging.info("Retraining already in progress, skipping...")
            return
        
        self.is_retraining = True
        self.last_retraining_time = datetime.now()
        
        # Start retraining in background thread
        self.retraining_thread = threading.Thread(target=self._retrain_model)
        self.retraining_thread.start()
        
        logging.info("Started background retraining process")
    
    def _retrain_model(self):
        """Retrain the AI model (runs in background thread)."""
        try:
            logging.info("Starting AI model retraining...")
            
            if not AI_OPTIMIZATION_AVAILABLE:
                logging.error("AI optimization not available for retraining")
                return
            
            # Initialize trainer
            trainer = AITrainer(
                config_path=self.config_path,
                optimization_bounds=self.optimization_bounds,
                save_dir=os.path.join(self.save_dir, f"retrain_{int(time.time())}")
            )
            
            # Load existing model if available
            if self.current_model_path and os.path.exists(self.current_model_path):
                trainer.load_checkpoint(self.current_model_path)
                logging.info(f"Loaded existing model for retraining: {self.current_model_path}")
            
            # Update environment with current market conditions
            self._update_trainer_environment(trainer)
            
            # Retrain with reduced episodes for faster adaptation
            retraining_episodes = 200
            training_summary = trainer.train(
                num_episodes=retraining_episodes,
                max_episode_steps=30,
                update_frequency=5,
                save_frequency=50,
                target_performance=0.1  # Stop early if good performance achieved
            )
            
            # Save the retrained model
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = os.path.join(self.save_dir, f"model_retrained_{model_timestamp}.pt")
            trainer.agent.save(new_model_path)
            
            # Update current model path
            self.current_model_path = new_model_path
            
            # Log retraining results
            best_performance = training_summary['performance_metrics']['best_global_performance']
            logging.info(f"Retraining completed! New model saved to: {new_model_path}")
            logging.info(f"Best performance achieved: {best_performance}")
            
            # Save retraining summary
            summary_path = os.path.join(self.save_dir, f"retrain_summary_{model_timestamp}.json")
            with open(summary_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'trigger_reason': 'performance_degradation',
                    'market_conditions': {k: list(v)[-5:] for k, v in self.market_conditions.items()},
                    'training_summary': training_summary,
                    'model_path': new_model_path
                }, f, indent=2, default=str)
        
        except Exception as e:
            logging.error(f"Error during retraining: {e}")
        
        finally:
            self.is_retraining = False
            logging.info("Retraining process completed")
    
    def _update_trainer_environment(self, trainer):
        """Update trainer environment with current market conditions."""
        # Calculate current market features
        if all(len(cond) >= 5 for cond in self.market_conditions.values()):
            market_features = {
                'volatility': np.mean(list(self.market_conditions['volatility'])[-5:]),
                'trend': np.mean(list(self.market_conditions['trend'])[-5:]),
                'volume': np.mean(list(self.market_conditions['volume'])[-5:]),
                'correlation': 0.0  # Would need price correlation data
            }
            
            # Update environment market features
            trainer.env.update_market_features(market_features)
            logging.info(f"Updated trainer environment with market features: {market_features}")
    
    async def _adjust_parameters_if_needed(self, current_metrics: PerformanceMetrics):
        """Make real-time micro-adjustments to parameters if needed."""
        # This would implement small, safe parameter adjustments based on
        # current market conditions without full retraining
        
        # For now, just log the concept
        if len(self.performance_history) >= 5:
            recent_performance = np.mean([m.gain for m in list(self.performance_history)[-5:]])
            
            if recent_performance < -0.05:  # Poor recent performance
                logging.debug("Considering micro-parameter adjustments due to poor recent performance")
                # Could implement small exposure adjustments, etc.
    
    def _log_monitoring_status(self, current_metrics: PerformanceMetrics):
        """Log current monitoring status."""
        if len(self.performance_history) % 24 == 0:  # Log every 24 cycles (daily if hourly monitoring)
            recent_metrics = list(self.performance_history)[-24:] if len(self.performance_history) >= 24 else list(self.performance_history)
            
            avg_gain = np.mean([m.gain for m in recent_metrics])
            avg_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics])
            max_drawdown = max([m.drawdown for m in recent_metrics])
            
            logging.info(f"AutoOptimizer Status - "
                        f"24h Avg Gain: {avg_gain:.3f}, "
                        f"24h Avg Sharpe: {avg_sharpe:.3f}, "
                        f"24h Max Drawdown: {max_drawdown:.3f}, "
                        f"Retraining: {'Yes' if self.is_retraining else 'No'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-optimizer status."""
        recent_metrics = list(self.performance_history)[-10:] if self.performance_history else []
        
        return {
            'is_running': self.is_running,
            'is_retraining': self.is_retraining,
            'last_retraining': self.last_retraining_time.isoformat() if self.last_retraining_time else None,
            'current_model': self.current_model_path,
            'performance_data_points': len(self.performance_history),
            'recent_avg_gain': np.mean([m.gain for m in recent_metrics]) if recent_metrics else 0,
            'recent_avg_sharpe': np.mean([m.sharpe_ratio for m in recent_metrics]) if recent_metrics else 0,
            'market_conditions': {
                k: list(v)[-5:] if v else [] 
                for k, v in self.market_conditions.items()
            }
        }
    
    def save_state(self):
        """Save auto-optimizer state to disk."""
        state_file = os.path.join(self.save_dir, "auto_optimizer_state.json")
        
        state = {
            'performance_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'balance': m.balance,
                    'gain': m.gain,
                    'drawdown': m.drawdown,
                    'sharpe_ratio': m.sharpe_ratio,
                    'volume': m.volume,
                    'num_trades': m.num_trades,
                    'win_rate': m.win_rate
                }
                for m in self.performance_history
            ],
            'last_retraining_time': self.last_retraining_time.isoformat() if self.last_retraining_time else None,
            'current_model_path': self.current_model_path,
            'market_conditions': {k: list(v) for k, v in self.market_conditions.items()}
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logging.info(f"Auto-optimizer state saved to {state_file}")
    
    def load_state(self):
        """Load auto-optimizer state from disk."""
        state_file = os.path.join(self.save_dir, "auto_optimizer_state.json")
        
        if not os.path.exists(state_file):
            logging.info("No saved state found")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore performance history
            for perf_data in state.get('performance_history', []):
                metrics = PerformanceMetrics(
                    timestamp=datetime.fromisoformat(perf_data['timestamp']),
                    balance=perf_data['balance'],
                    gain=perf_data['gain'],
                    drawdown=perf_data['drawdown'],
                    sharpe_ratio=perf_data['sharpe_ratio'],
                    volume=perf_data['volume'],
                    num_trades=perf_data['num_trades'],
                    win_rate=perf_data['win_rate']
                )
                self.performance_history.append(metrics)
            
            # Restore other state
            if state.get('last_retraining_time'):
                self.last_retraining_time = datetime.fromisoformat(state['last_retraining_time'])
            
            self.current_model_path = state.get('current_model_path')
            
            # Restore market conditions
            for condition_name, condition_data in state.get('market_conditions', {}).items():
                if condition_name in self.market_conditions:
                    self.market_conditions[condition_name].extend(condition_data)
            
            logging.info(f"Auto-optimizer state loaded from {state_file}")
            logging.info(f"Restored {len(self.performance_history)} performance data points")
        
        except Exception as e:
            logging.error(f"Error loading auto-optimizer state: {e}")


# Example usage and integration
async def run_auto_optimizer_example():
    """Example of how to use the AutoOptimizer."""
    # Define optimization bounds (example)
    bounds = [
        (0.1, 2.0),   # total_wallet_exposure_limit
        (0.001, 0.1), # entry_initial_qty_pct
        (1.0, 10.0),  # entry_initial_eprice_ema_dist
        # ... add more parameter bounds
    ]
    
    # Initialize auto-optimizer
    auto_optimizer = AutoOptimizer(
        config_path="configs/template.json",
        optimization_bounds=bounds,
        monitoring_interval=3600,  # Check every hour
        retraining_threshold=-0.1,  # Retrain if performance drops 10%
        min_retraining_interval=86400  # At least 24 hours between retrainings
    )
    
    # Load previous state if available
    auto_optimizer.load_state()
    
    try:
        # Start monitoring
        await auto_optimizer.start_monitoring()
    except KeyboardInterrupt:
        logging.info("Shutting down auto-optimizer...")
    finally:
        # Save state and stop
        auto_optimizer.save_state()
        auto_optimizer.stop_monitoring()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_auto_optimizer_example())