"""
Trading environment wrapper for Passivbot AI optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config_utils import load_config, format_config
from backtest import prep_backtest_args
from optimize import individual_to_config, config_to_individual
from pure_funcs import calc_hash, denumpyize


class PassivbotEnvironment:
    """
    Environment wrapper that interfaces with Passivbot's optimization system.
    Converts optimization problems into RL state-action-reward framework.
    """

    def __init__(
        self,
        config_path: str,
        optimization_bounds: List[Tuple[float, float]],
        sig_digits: int = 4,
        max_episode_steps: int = 50,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the environment.

        Args:
            config_path: Path to base configuration file
            optimization_bounds: List of (min, max) bounds for each parameter
            sig_digits: Significant digits for parameter rounding
            max_episode_steps: Maximum steps per episode
            reward_weights: Weights for different reward components
        """
        self.config_path = config_path
        self.bounds = optimization_bounds
        self.sig_digits = sig_digits
        self.max_episode_steps = max_episode_steps

        # Load base configuration
        self.base_config = load_config(config_path)
        self.base_config = format_config(self.base_config)

        # Reward configuration
        self.reward_weights = reward_weights or {
            "gain": 0.3,
            "sharpe_ratio": 0.2,
            "drawdown_worst": -0.2,
            "calmar_ratio": 0.15,
            "volume_pct_per_day_avg": 0.1,
            "position_held_hours_mean": -0.05,
        }

        # Environment state
        self.current_step = 0
        self.episode_history = []
        self.best_performance = None
        self.current_individual = None
        self.state_history = []

        # Market condition features (to be updated with real market data)
        self.market_features = {
            "volatility": 0.0,
            "trend": 0.0,
            "volume": 0.0,
            "correlation": 0.0,
        }

        # State normalization parameters
        self.state_mean = None
        self.state_std = None

        logging.info(f"Environment initialized with {len(self.bounds)} parameters")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.episode_history = []

        # Generate random initial configuration within bounds
        self.current_individual = []
        for low, high in self.bounds:
            if high == low:
                self.current_individual.append(low)
            else:
                value = np.random.uniform(low, high)
                # Round to significant digits
                if value != 0:
                    exp = np.floor(np.log10(np.abs(value))) - (self.sig_digits - 1)
                    value = np.round(value, -int(exp))
                self.current_individual.append(value)

        self.current_individual = np.array(self.current_individual)

        # Get initial state
        state = self._get_state()
        self.state_history = [state]

        logging.debug(
            f"Environment reset. Initial individual: {self.current_individual[:5]}..."
        )
        return state

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to take (parameter modifications)

        Returns:
            next_state: Next state observation
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1

        # Apply action to current individual
        new_individual = self._apply_action(self.current_individual, action)

        # Evaluate the new configuration
        performance = self._evaluate_individual(new_individual)

        # Calculate reward
        reward = self._calculate_reward(performance)

        # Update state
        self.current_individual = new_individual
        next_state = self._get_state()
        self.state_history.append(next_state)

        # Store episode history
        self.episode_history.append(
            {
                "step": self.current_step,
                "individual": new_individual.copy(),
                "performance": performance,
                "reward": reward,
                "action": action.copy(),
            }
        )

        # Check if episode is done
        done = (
            self.current_step >= self.max_episode_steps
            or self._is_converged()
            or self._is_stuck()
        )

        # Update best performance
        if self.best_performance is None or self._is_better_performance(performance):
            self.best_performance = performance

        info = {
            "performance": performance,
            "best_performance": self.best_performance,
            "episode_step": self.current_step,
            "convergence": self._check_convergence(),
        }

        logging.debug(
            f"Step {self.current_step}: reward={reward:.4f}, performance={performance.get('gain', 0):.4f}"
        )

        return next_state, reward, done, info

    def _apply_action(self, individual: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Apply action to modify individual parameters.

        Args:
            individual: Current parameter values
            action: Modification actions

        Returns:
            Modified parameter values
        """
        # Actions are relative modifications (e.g., multiply by (1 + action))
        # Clip actions to reasonable range
        clipped_action = np.clip(action, -0.1, 0.1)

        # Apply modifications
        new_individual = individual * (1 + clipped_action)

        # Enforce bounds
        for i, (low, high) in enumerate(self.bounds):
            new_individual[i] = np.clip(new_individual[i], low, high)

            # Round to significant digits
            if new_individual[i] != 0 and high != low:
                exp = np.floor(np.log10(np.abs(new_individual[i]))) - (
                    self.sig_digits - 1
                )
                new_individual[i] = np.round(new_individual[i], -int(exp))

        return new_individual

    def _evaluate_individual(self, individual: np.ndarray) -> Dict[str, float]:
        """
        Evaluate trading performance for given individual.

        Args:
            individual: Parameter configuration to evaluate

        Returns:
            Performance metrics dictionary
        """
        try:
            # Convert individual to config format
            # Use lambda for optimizer_overrides since it expects a function
            config = individual_to_config(
                individual,
                lambda overrides_list, config, pside: config,  # No-op optimizer_overrides
                [],  # overrides_list (empty for now)
                self.base_config,
            )

            # Run backtest
            # Note: This is a simplified version - in practice you'd want to
            # interface with the actual backtest system
            performance = self._run_simplified_backtest(config)

            return performance

        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            # Return poor performance if evaluation fails
            return {
                "gain": -1.0,
                "sharpe_ratio": -2.0,
                "drawdown_worst": 0.9,
                "calmar_ratio": -1.0,
                "volume_pct_per_day_avg": 0.0,
                "position_held_hours_mean": 1000.0,
            }

    def _run_simplified_backtest(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run simplified backtest for fast evaluation.
        In practice, this would interface with the actual Rust backtest engine.

        Args:
            config: Trading configuration

        Returns:
            Performance metrics
        """
        # This is a placeholder - replace with actual backtest call
        # For now, return random performance with some correlation to parameters

        # Extract some key parameters
        try:
            long_exposure = (
                config.get("bot", {})
                .get("long", {})
                .get("total_wallet_exposure_limit", 1.0)
            )
            short_exposure = (
                config.get("bot", {})
                .get("short", {})
                .get("total_wallet_exposure_limit", 1.0)
            )

            # Simple heuristic for demonstration
            avg_exposure = (long_exposure + short_exposure) / 2

            # Random performance with bias based on exposure
            base_gain = np.random.normal(0.1, 0.3)
            exposure_penalty = max(0, avg_exposure - 0.5) * 0.5

            performance = {
                "gain": base_gain - exposure_penalty,
                "sharpe_ratio": np.random.normal(1.0, 0.5),
                "drawdown_worst": np.random.uniform(0.05, 0.3),
                "calmar_ratio": np.random.normal(0.5, 0.3),
                "volume_pct_per_day_avg": np.random.uniform(0.1, 2.0),
                "position_held_hours_mean": np.random.uniform(1, 100),
            }

        except Exception as e:
            logging.error(f"Error in simplified backtest: {e}")
            performance = {
                "gain": -0.5,
                "sharpe_ratio": -1.0,
                "drawdown_worst": 0.5,
                "calmar_ratio": -0.5,
                "volume_pct_per_day_avg": 0.0,
                "position_held_hours_mean": 500.0,
            }

        return performance

    def _calculate_reward(self, performance: Dict[str, float]) -> float:
        """
        Calculate reward based on performance metrics.

        Args:
            performance: Performance metrics

        Returns:
            Scalar reward value
        """
        reward = 0.0

        for metric, weight in self.reward_weights.items():
            if metric in performance:
                value = performance[metric]

                # Normalize different metrics to similar scales
                if metric in ["gain", "sharpe_ratio", "calmar_ratio"]:
                    # Higher is better
                    normalized_value = np.tanh(value)
                elif metric == "drawdown_worst":
                    # Lower is better (negative drawdown)
                    normalized_value = -np.tanh(value * 5)  # Scale up for sensitivity
                elif metric == "volume_pct_per_day_avg":
                    # Target around 1.0
                    normalized_value = -abs(value - 1.0)
                elif metric == "position_held_hours_mean":
                    # Target around 24 hours
                    normalized_value = -abs(value - 24) / 24
                else:
                    normalized_value = np.tanh(value)

                reward += weight * normalized_value

        # Add exploration bonus for trying new parameter regions
        exploration_bonus = self._calculate_exploration_bonus()
        reward += 0.01 * exploration_bonus

        # Add stability penalty for excessive parameter changes
        stability_penalty = self._calculate_stability_penalty()
        reward -= 0.005 * stability_penalty

        return reward

    def _calculate_exploration_bonus(self) -> float:
        """Calculate bonus for exploring new parameter regions."""
        if len(self.episode_history) < 2:
            return 1.0

        # Measure diversity of recent configurations
        recent_individuals = [
            entry["individual"] for entry in self.episode_history[-5:]
        ]

        if len(recent_individuals) < 2:
            return 1.0

        # Calculate average pairwise distance
        distances = []
        for i in range(len(recent_individuals)):
            for j in range(i + 1, len(recent_individuals)):
                dist = np.linalg.norm(recent_individuals[i] - recent_individuals[j])
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0
        return np.tanh(avg_distance)

    def _calculate_stability_penalty(self) -> float:
        """Calculate penalty for excessive parameter instability."""
        if len(self.episode_history) < 3:
            return 0.0

        # Look at parameter changes over last few steps
        recent_individuals = [
            entry["individual"] for entry in self.episode_history[-3:]
        ]

        # Calculate variance in parameter changes
        changes = []
        for i in range(1, len(recent_individuals)):
            change = np.abs(recent_individuals[i] - recent_individuals[i - 1])
            changes.append(change)

        if len(changes) == 0:
            return 0.0

        avg_change = np.mean(changes)
        return np.tanh(avg_change * 10)  # Scale up for sensitivity

    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.

        Returns:
            State vector
        """
        # Current parameter values (normalized)
        param_state = self.current_individual.copy()

        # Performance history features
        if len(self.episode_history) > 0:
            recent_performance = [
                entry["performance"] for entry in self.episode_history[-5:]
            ]
            performance_features = self._extract_performance_features(
                recent_performance
            )
        else:
            performance_features = np.zeros(6)  # Number of performance metrics

        # Market condition features
        market_state = np.array(
            [
                self.market_features["volatility"],
                self.market_features["trend"],
                self.market_features["volume"],
                self.market_features["correlation"],
            ]
        )

        # Episode progress
        progress_features = np.array(
            [
                self.current_step / self.max_episode_steps,
                len(self.episode_history) / self.max_episode_steps,
            ]
        )

        # Combine all features
        state = np.concatenate(
            [param_state, performance_features, market_state, progress_features]
        )

        return state.astype(np.float32)

    def _extract_performance_features(
        self, performance_history: List[Dict[str, float]]
    ) -> np.ndarray:
        """Extract features from performance history."""
        if not performance_history:
            return np.zeros(6)

        # Get latest performance
        latest = performance_history[-1]

        features = [
            latest.get("gain", 0.0),
            latest.get("sharpe_ratio", 0.0),
            latest.get("drawdown_worst", 0.0),
            latest.get("calmar_ratio", 0.0),
            latest.get("volume_pct_per_day_avg", 0.0),
            latest.get("position_held_hours_mean", 0.0),
        ]

        return np.array(features, dtype=np.float32)

    def _is_better_performance(self, performance: Dict[str, float]) -> bool:
        """Check if performance is better than current best."""
        if self.best_performance is None:
            return True

        # Use weighted sum for comparison
        current_score = sum(
            self.reward_weights.get(k, 0) * v for k, v in performance.items()
        )
        best_score = sum(
            self.reward_weights.get(k, 0) * v for k, v in self.best_performance.items()
        )

        return current_score > best_score

    def _is_converged(self) -> bool:
        """Check if optimization has converged."""
        if len(self.episode_history) < 10:
            return False

        # Check if performance has plateaued
        recent_rewards = [entry["reward"] for entry in self.episode_history[-10:]]
        reward_std = np.std(recent_rewards)

        return reward_std < 0.01  # Very low variance indicates convergence

    def _is_stuck(self) -> bool:
        """Check if optimization is stuck in local minimum."""
        if len(self.episode_history) < 20:
            return False

        # Check if we're not improving over a longer period
        recent_rewards = [entry["reward"] for entry in self.episode_history[-20:]]
        early_avg = np.mean(recent_rewards[:10])
        late_avg = np.mean(recent_rewards[10:])

        # Stuck if no improvement over 20 steps
        return late_avg <= early_avg + 0.005

    def _check_convergence(self) -> Dict[str, float]:
        """Get convergence metrics."""
        if len(self.episode_history) < 5:
            return {"reward_std": 1.0, "param_std": 1.0}

        recent_rewards = [entry["reward"] for entry in self.episode_history[-5:]]
        recent_individuals = [
            entry["individual"] for entry in self.episode_history[-5:]
        ]

        reward_std = np.std(recent_rewards)
        param_std = np.mean(
            [
                np.std([ind[i] for ind in recent_individuals])
                for i in range(len(recent_individuals[0]))
            ]
        )

        return {"reward_std": reward_std, "param_std": param_std}

    def update_market_features(self, features: Dict[str, float]):
        """Update market condition features."""
        self.market_features.update(features)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode."""
        if not self.episode_history:
            return {}

        rewards = [entry["reward"] for entry in self.episode_history]
        performances = [entry["performance"] for entry in self.episode_history]

        return {
            "total_steps": len(self.episode_history),
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "final_reward": rewards[-1] if rewards else 0,
            "best_performance": self.best_performance,
            "final_performance": performances[-1] if performances else {},
            "convergence_metrics": self._check_convergence(),
        }
