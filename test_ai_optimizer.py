#!/usr/bin/env python3
"""
Test script for the AI optimization system.
Tests the GPU-accelerated AI learning functionality.
"""

import sys
import os
import logging
import asyncio
import argparse
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_ai_imports():
    """Test that AI modules can be imported."""
    print("Testing AI module imports...")
    
    try:
        from ai_optimizer import AITrainer, PPOAgent, PassivbotEnvironment, setup_gpu
        print("✓ Successfully imported AI optimization modules")
        return True
    except ImportError as e:
        print(f"✗ Failed to import AI modules: {e}")
        print("Make sure PyTorch dependencies are installed: pip install -r requirements.txt")
        return False

def test_gpu_setup():
    """Test GPU setup and availability."""
    print("\nTesting GPU setup...")
    
    try:
        from ai_optimizer.utils import setup_gpu, get_device
        
        device = get_device()
        gpu_config = setup_gpu()
        
        print(f"✓ Device: {device}")
        print(f"✓ GPU config: {gpu_config}")
        
        if device.type == 'cuda':
            print("✓ CUDA GPU detected and available")
        elif device.type == 'mps':
            print("✓ Apple Metal Performance Shaders available")
        else:
            print("✓ Using CPU (no GPU acceleration)")
        
        return True
    except Exception as e:
        print(f"✗ GPU setup failed: {e}")
        return False

def test_environment():
    """Test the trading environment wrapper."""
    print("\nTesting PassivbotEnvironment...")
    
    try:
        from ai_optimizer import PassivbotEnvironment
        
        # Generate proper bounds for all parameters in template config
        from config_utils import load_config
        from optimize import get_bound_keys_ignored
        
        config = load_config("configs/template.json")
        keys_ignored = get_bound_keys_ignored()
        
        bounds = []
        for pside in sorted(config["bot"]):
            for key in sorted(config["bot"][pside]):
                if key not in keys_ignored:
                    val = config["bot"][pside][key]
                    if isinstance(val, bool):
                        bounds.append((0, 1))  # Boolean as 0-1
                    elif isinstance(val, int):
                        bounds.append((max(1, int(val * 0.5)), int(val * 2)))
                    else:  # float
                        if val == 0:
                            bounds.append((-0.1, 0.1))
                        elif val > 0:
                            bounds.append((val * 0.1, val * 3.0))
                        else:  # negative
                            bounds.append((val * 3.0, val * 0.1))
        
        print(f"Generated {len(bounds)} bounds for all parameters")
        
        # Test environment creation
        env = PassivbotEnvironment(
            config_path="configs/template.json",
            optimization_bounds=bounds,
            max_episode_steps=10  # Short episode for testing
        )
        
        print(f"✓ Environment created with {len(bounds)} parameters")
        
        # Test environment reset
        state = env.reset()
        print(f"✓ Environment reset, state shape: {state.shape}")
        
        # Test environment step
        action = np.random.uniform(-0.1, 0.1, len(bounds))
        next_state, reward, done, info = env.step(action)
        
        print(f"✓ Environment step completed")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Done: {done}")
        print(f"  - Next state shape: {next_state.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_neural_networks():
    """Test neural network architectures."""
    print("\nTesting neural networks...")
    
    try:
        from ai_optimizer.neural_networks import PolicyNetwork, ValueNetwork
        import torch
        
        state_dim = 20
        action_dim = 5
        batch_size = 4
        
        # Test policy network
        policy_net = PolicyNetwork(state_dim, action_dim)
        test_state = torch.randn(batch_size, state_dim)
        
        action_mean, action_std = policy_net(test_state)
        print(f"✓ Policy network: input {test_state.shape} -> output {action_mean.shape}, {action_std.shape}")
        
        # Test value network
        value_net = ValueNetwork(state_dim)
        value = value_net(test_state)
        print(f"✓ Value network: input {test_state.shape} -> output {value.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_ppo_agent():
    """Test PPO agent functionality."""
    print("\nTesting PPO agent...")
    
    try:
        from ai_optimizer import PPOAgent
        import torch
        
        state_dim = 20
        action_dim = 5
        
        # Create agent
        agent = PPOAgent(state_dim, action_dim)
        print(f"✓ PPO agent created (state_dim={state_dim}, action_dim={action_dim})")
        
        # Test action selection
        test_state = np.random.randn(state_dim)
        action, value, log_prob = agent.get_action_and_value(test_state)
        
        print(f"✓ Action selection: action shape {action.shape}, value {value:.4f}")
        
        # Test transition storage
        agent.store_transition(test_state, action, 0.1, value, log_prob, False)
        print("✓ Transition storage works")
        
        return True
    except Exception as e:
        print(f"✗ PPO agent test failed: {e}")
        return False

def test_training_integration():
    """Test training integration."""
    print("\nTesting training integration...")
    
    try:
        from ai_optimizer import AITrainer
        
        # Simple bounds for testing
        bounds = [(0.1, 2.0), (0.001, 0.1), (1.0, 10.0)]
        
        # Create trainer
        trainer = AITrainer(
            config_path="configs/template.json",
            optimization_bounds=bounds,
            save_dir="test_models",
            log_dir="test_logs"
        )
        
        print(f"✓ AITrainer created")
        print(f"  - State dimension: {trainer.state_dim}")
        print(f"  - Action dimension: {trainer.action_dim}")
        print(f"  - Device: {trainer.device}")
        
        # Test short training run
        print("Running short training test (5 episodes)...")
        summary = trainer.train(
            num_episodes=5,
            max_episode_steps=10,
            update_frequency=2,
            save_frequency=10
        )
        
        print("✓ Training completed successfully")
        print(f"  - Best performance: {summary['performance_metrics']['best_global_performance']}")
        
        return True
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")
        return False

def test_optimization_modes():
    """Test different optimization modes."""
    print("\nTesting optimization modes...")
    
    try:
        # Test that the optimization functions exist and can be imported
        import src.optimize as opt_module
        
        # Check if AI functions are available
        if hasattr(opt_module, 'run_ai_optimization'):
            print("✓ AI optimization mode available")
        else:
            print("✗ AI optimization mode not found")
            return False
        
        if hasattr(opt_module, 'run_hybrid_optimization'):
            print("✓ Hybrid optimization mode available")
        else:
            print("✗ Hybrid optimization mode not found")
            return False
        
        print("✓ All optimization modes available")
        return True
    except Exception as e:
        print(f"✗ Optimization modes test failed: {e}")
        return False

def test_auto_optimizer():
    """Test auto-optimization controller."""
    print("\nTesting auto-optimizer...")
    
    try:
        from src.auto_optimizer import AutoOptimizer, PerformanceMetrics
        from datetime import datetime
        
        bounds = [(0.1, 2.0), (0.001, 0.1)]
        
        auto_opt = AutoOptimizer(
            config_path="configs/template.json",
            optimization_bounds=bounds,
            monitoring_interval=1,  # 1 second for testing
            save_dir="test_auto_optimizer"
        )
        
        print("✓ AutoOptimizer created")
        
        # Test performance metrics
        test_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            balance=10000.0,
            gain=0.05,
            drawdown=0.02,
            sharpe_ratio=1.5,
            volume=500.0,
            num_trades=10,
            win_rate=0.6
        )
        
        auto_opt.performance_history.append(test_metrics)
        status = auto_opt.get_status()
        
        print("✓ Performance tracking works")
        print(f"  - Status: {status['is_running']}")
        print(f"  - Data points: {status['performance_data_points']}")
        
        return True
    except Exception as e:
        print(f"✗ Auto-optimizer test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting AI Optimization System Tests")
    print("=" * 50)
    
    tests = [
        test_ai_imports,
        test_gpu_setup,
        test_environment,
        test_neural_networks,
        test_ppo_agent,
        test_training_integration,
        test_optimization_modes,
        test_auto_optimizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"⚠️  Test failed: {test.__name__}")
        except Exception as e:
            print(f"💥 Test crashed: {test.__name__} - {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI optimization system is ready to use.")
        print("\nUsage examples:")
        print("1. AI-only optimization:")
        print("   python src/optimize.py --ai-mode --ai-episodes 500")
        print("\n2. Hybrid optimization:")
        print("   python src/optimize.py --hybrid-mode --ai-episodes 200")
        print("\n3. Auto-optimization (live trading):")
        print("   python src/auto_optimizer.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Common fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check PyTorch installation: pip install torch torchrl gymnasium")
        print("- Ensure configs/template.json exists")
    
    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test AI optimization system')
    parser.add_argument('--check-deps-only', action='store_true', 
                       help='Only check if AI dependencies are available')
    args = parser.parse_args()
    
    if args.check_deps_only:
        # Simple dependency check
        try:
            from ai_optimizer import AITrainer, PPOAgent, PassivbotEnvironment, setup_gpu
            print("AI dependencies are installed and working")
            sys.exit(0)
        except ImportError as e:
            print(f"AI dependency check failed: {e}")
            sys.exit(1)
    
    # Run full tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)