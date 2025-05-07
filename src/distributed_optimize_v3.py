#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import json
import logging
import socket
import time
import uuid
import zmq
import zmq.asyncio
import numpy as np
import psutil
import tempfile
import multiprocessing
import platform
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
import zlib
import tqdm
import datetime


# Import from passivbot
from optimize import (
    manage_rust_compilation,
    prepare_hlcvs_mss,
    create_shared_memory_file,
    individual_to_config,
    optimizer_overrides,
    validate_array,
    check_disk_space,
    calc_hash,
    round_floats,
)
from procedures import (
    load_config,
    format_config,
    make_get_filepath,
    utc_ms,
)
from pure_funcs import (
    ts_to_date_utc,
    date_to_ts,
    get_template_live_config,
)
from downloader import add_all_eligible_coins_to_config
from pareto_store import ParetoStore
from opt_utils import make_json_serializable, dominates

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Set process priority if possible
try:
    os.nice(10)  # Lower priority
except Exception:
    pass


def dummy_task(x):
    # Do some meaningless computation
    result = 0
    for i in range(1000000):
        result += i
    return result


def process_with_cached_evaluator(
    individual,
    overrides_list,
    config,
    shared_memory_files,
    hlcvs_shapes,
    hlcvs_dtypes,
    btc_usd_shared_memory_files,
    btc_usd_dtypes,
    msss,
    worker_id,
):
    """Process an individual with a cached evaluator (module-level function for pickling)"""
    # Try to optimize this process for CPU usage
    try:
        # Get current process
        process = psutil.Process()

        # Set CPU affinity if supported
        if hasattr(process, "cpu_affinity"):
            try:
                # Calculate which core this worker should use
                core_id = worker_id % psutil.cpu_count()
                process.cpu_affinity([core_id])
            except Exception:
                pass

        # Set process priority
        if sys.platform == "win32":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            # Unix-like systems
            process.nice(10)  # Lower priority
    except Exception:
        pass

    from optimize import Evaluator, individual_to_config, optimizer_overrides
    import multiprocessing

    # Create a queue for this process
    queue = multiprocessing.Queue()

    # Create evaluator
    evaluator = Evaluator(
        shared_memory_files=shared_memory_files,
        hlcvs_shapes=hlcvs_shapes,
        hlcvs_dtypes=hlcvs_dtypes,
        btc_usd_shared_memory_files=btc_usd_shared_memory_files,
        btc_usd_dtypes=btc_usd_dtypes,
        msss=msss,
        config=config,
        results_queue=queue,
        seen_hashes={},
        duplicate_counter={"count": 0},
    )

    # Evaluate the individual
    objectives = evaluator.evaluate(individual, overrides_list)

    # Get result from queue
    result = queue.get()

    # Add individual to result
    result["individual"] = individual

    # Ensure config is present
    if "config" not in result:
        result["config"] = individual_to_config(
            individual, optimizer_overrides, overrides_list, template=config
        )

    return result


class CachedEvaluator:
    """Caches the evaluator to prevent repeated initialization"""

    def __init__(
        self,
        shared_memory_files,
        hlcvs_shapes,
        hlcvs_dtypes,
        btc_usd_shared_memory_files,
        btc_usd_dtypes,
        msss,
        config,
    ):
        self.shared_memory_files = shared_memory_files
        self.hlcvs_shapes = hlcvs_shapes
        self.hlcvs_dtypes = hlcvs_dtypes
        self.btc_usd_shared_memory_files = btc_usd_shared_memory_files
        self.btc_usd_dtypes = btc_usd_dtypes
        self.msss = msss
        self.config = config
        self.evaluator_cache = {}
        self.seen_hashes = {}
        self.duplicate_counter = {"count": 0}

    def get_evaluator(self):
        """Get or create an evaluator instance"""
        from optimize import Evaluator
        import multiprocessing

        # Create a queue for this process
        queue = multiprocessing.Queue()

        # Create evaluator if not already cached
        evaluator = Evaluator(
            shared_memory_files=self.shared_memory_files,
            hlcvs_shapes=self.hlcvs_shapes,
            hlcvs_dtypes=self.hlcvs_dtypes,
            btc_usd_shared_memory_files=self.btc_usd_shared_memory_files,
            btc_usd_dtypes=self.btc_usd_dtypes,
            msss=self.msss,
            config=self.config,
            results_queue=queue,
            seen_hashes=self.seen_hashes,
            duplicate_counter=self.duplicate_counter,
        )

        return evaluator, queue


# Wrapper function for evaluating individuals in separate processes
def evaluate_individual_wrapper(
    individual,
    overrides_list,
    config,
    shared_memory_files,
    hlcvs_shapes,
    hlcvs_dtypes,
    btc_usd_shared_memory_files,
    btc_usd_dtypes,
    msss,
):
    """
    Wrapper function for evaluating individuals in separate processes.
    This needs to be at module level to be picklable for multiprocessing.
    """
    from optimize import Evaluator
    import multiprocessing

    # Create a queue for this process
    queue = multiprocessing.Queue()

    # Create evaluator
    evaluator = Evaluator(
        shared_memory_files=shared_memory_files,
        hlcvs_shapes=hlcvs_shapes,
        hlcvs_dtypes=hlcvs_dtypes,
        btc_usd_shared_memory_files=btc_usd_shared_memory_files,
        btc_usd_dtypes=btc_usd_dtypes,
        msss=msss,
        config=config,
        results_queue=queue,
        seen_hashes={},
        duplicate_counter={"count": 0},
    )

    # Evaluate the individual
    objectives = evaluator.evaluate(individual, overrides_list)

    # Get result from queue
    result = queue.get()

    # Add individual to result
    result["individual"] = individual

    # Ensure config is present
    if "config" not in result:
        from optimize import individual_to_config, optimizer_overrides

        result["config"] = individual_to_config(
            individual, optimizer_overrides, overrides_list, template=config
        )

    return result


# GPU-accelerated evaluation functions
def evaluate_batch_gpu(
    individuals,
    overrides_list,
    config,
    market_data,
    gpu_type="auto",
    gpu_id=0,
    max_batch_size=64,
):
    """
    Evaluate a batch of individuals using GPU acceleration.
    Supports both NVIDIA (via CuPy) and AMD (via PyOpenCL) GPUs.

    Args:
        individuals: List of individuals to evaluate
        overrides_list: List of overrides
        config: Configuration dictionary
        market_data: Market data dictionary
        gpu_type: "nvidia", "amd", or "auto" for automatic detection
        gpu_id: GPU device ID to use
        max_batch_size: Maximum batch size to process at once

    Returns:
        List of evaluation results
    """
    # Detect GPU type if set to auto
    if gpu_type == "auto":
        gpu_type = detect_gpu_type()

    if gpu_type == "none":
        # No GPU available, fall back to CPU
        logging.warning("No GPU detected, falling back to CPU evaluation")
        return None

    try:
        if gpu_type == "nvidia":
            return evaluate_batch_nvidia(
                individuals, overrides_list, config, market_data, gpu_id, max_batch_size
            )
        elif gpu_type == "amd":
            return evaluate_batch_amd(
                individuals, overrides_list, config, market_data, gpu_id, max_batch_size
            )
        else:
            logging.warning(f"Unknown GPU type: {gpu_type}, falling back to CPU")
            return None
    except Exception as e:
        logging.error(f"GPU evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def detect_gpu_type():
    """
    Detect available GPU type (NVIDIA or AMD)

    Returns:
        "nvidia", "amd", or "none" if no GPU is detected
    """
    # First try to detect NVIDIA GPU via CuPy
    try:
        import cupy as cp

        # Try to initialize CuPy
        cp.cuda.runtime.getDeviceCount()
        return "nvidia"
    except (ImportError, Exception):
        pass

    # Then try to detect AMD GPU via PyOpenCL
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        for platform in platforms:
            if "AMD" in platform.name:
                return "amd"
    except (ImportError, Exception):
        pass

    # Try to detect GPU via system commands
    try:
        if sys.platform == "linux":
            # Check for NVIDIA GPU
            nvidia_output = os.popen("lspci | grep -i nvidia").read()
            if "nvidia" in nvidia_output.lower():
                return "nvidia"

            # Check for AMD GPU
            amd_output = os.popen("lspci | grep -i amd").read()
            if "amd" in amd_output.lower() and "vga" in amd_output.lower():
                return "amd"
        elif sys.platform == "win32":
            # On Windows, check via wmic
            gpu_output = os.popen("wmic path win32_VideoController get name").read()
            if "nvidia" in gpu_output.lower():
                return "nvidia"
            elif "amd" in gpu_output.lower() or "radeon" in gpu_output.lower():
                return "amd"
    except Exception:
        pass

    return "none"


def evaluate_batch_nvidia(
    individuals, overrides_list, config, market_data, gpu_id=0, max_batch_size=64
):
    """
    Evaluate a batch of individuals using NVIDIA GPU via CuPy
    """
    try:
        import cupy as cp

        # Set GPU device
        cp.cuda.Device(gpu_id).use()

        results = []

        # Process in batches to avoid GPU memory issues
        for i in range(0, len(individuals), max_batch_size):
            batch = individuals[i : i + max_batch_size]

            # Convert market data to GPU arrays
            gpu_data = {}
            for exchange, data in market_data.items():
                gpu_data[exchange] = cp.asarray(data)

            # Process batch
            batch_results = []
            for individual in batch:
                # Convert individual to parameters
                params = individual_to_config(
                    individual, optimizer_overrides, overrides_list, template=config
                )

                # Run backtest on GPU
                # This is a simplified placeholder - actual implementation would
                # need to port the backtest logic to CUDA
                result = run_backtest_on_gpu_nvidia(params, gpu_data, config)
                batch_results.append(result)

            # Add batch results
            results.extend(batch_results)

            # Free GPU memory
            for exchange in gpu_data:
                gpu_data[exchange] = None
            cp.get_default_memory_pool().free_all_blocks()

        return results

    except Exception as e:
        logging.error(f"NVIDIA GPU evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def evaluate_batch_amd(
    individuals, overrides_list, config, market_data, gpu_id=0, max_batch_size=64
):
    """
    Evaluate a batch of individuals using AMD GPU via PyOpenCL
    """
    try:
        import pyopencl as cl
        import pyopencl.array
        import numpy as np

        # Initialize OpenCL
        platforms = cl.get_platforms()
        amd_platform = None
        for platform in platforms:
            if "AMD" in platform.name:
                amd_platform = platform
                break

        if amd_platform is None:
            logging.error("No AMD platform found")
            return None

        # Get devices and create context
        devices = amd_platform.get_devices()
        if gpu_id >= len(devices):
            logging.error(f"GPU ID {gpu_id} out of range, max is {len(devices)-1}")
            return None

        device = devices[gpu_id]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        results = []

        # Process in batches to avoid GPU memory issues
        for i in range(0, len(individuals), max_batch_size):
            batch = individuals[i : i + max_batch_size]

            # Convert market data to GPU arrays
            gpu_data = {}
            for exchange, data in market_data.items():
                gpu_data[exchange] = cl.array.to_device(queue, data)

            # Process batch
            batch_results = []
            for individual in batch:
                # Convert individual to parameters
                params = individual_to_config(
                    individual, optimizer_overrides, overrides_list, template=config
                )

                # Run backtest on GPU
                # This is a simplified placeholder - actual implementation would
                # need to port the backtest logic to OpenCL
                result = run_backtest_on_gpu_amd(params, gpu_data, ctx, queue, config)
                batch_results.append(result)

            # Add batch results
            results.extend(batch_results)

            # Free GPU memory
            for exchange in gpu_data:
                gpu_data[exchange] = None

        return results

    except Exception as e:
        logging.error(f"AMD GPU evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_backtest_on_gpu_nvidia(params, gpu_data, config):
    """
    Placeholder for actual NVIDIA GPU backtest implementation

    In a real implementation, this would:
    1. Compile and run CUDA kernels for the backtest logic
    2. Process the market data on the GPU
    3. Return the backtest results

    For now, we'll just return a dummy result
    """
    import cupy as cp

    # This is where you would implement the actual GPU backtest
    # For now, just return a dummy result
    logging.info("NVIDIA GPU backtest placeholder - not actually running on GPU yet")

    # Create a dummy result
    result = {
        "config": params,
        "individual": None,  # This would be filled in by the caller
        "analyses_combined": {
            "w_adg": float(cp.random.random()),
            "w_drawdown_worst": float(cp.random.random() * 0.2),
            "w_sharpe_ratio": float(cp.random.random() * 3),
        },
    }

    return result


def run_backtest_on_gpu_amd(params, gpu_data, ctx, queue, config):
    """
    Placeholder for actual AMD GPU backtest implementation

    In a real implementation, this would:
    1. Compile and run OpenCL kernels for the backtest logic
    2. Process the market data on the GPU
    3. Return the backtest results

    For now, we'll just return a dummy result
    """
    import numpy as np

    # This is where you would implement the actual GPU backtest
    # For now, just return a dummy result
    logging.info("AMD GPU backtest placeholder - not actually running on GPU yet")

    # Create a dummy result
    result = {
        "config": params,
        "individual": None,  # This would be filled in by the caller
        "analyses_combined": {
            "w_adg": float(np.random.random()),
            "w_drawdown_worst": float(np.random.random() * 0.2),
            "w_sharpe_ratio": float(np.random.random() * 3),
        },
    }

    return result


async def get_all_eligible_coins(exchanges):
    """
    Get all eligible coins from the specified exchanges.
    This is a fallback method if add_all_eligible_coins_to_config fails.
    """
    import ccxt.async_support as ccxt

    all_coins = set()

    for exchange_name in exchanges:
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class()

            # Fetch markets
            markets = await exchange.fetch_markets()

            # Extract coin symbols
            for market in markets:
                if market.get("type") == "swap" and market.get("active"):
                    base = market.get("base", "")
                    if "/" in base:
                        base = base.split("/")[0]
                    if base:
                        all_coins.add(base)

            # Close exchange
            await exchange.close()

        except Exception as e:
            logging.error(f"Error fetching markets from {exchange_name}: {e}")

    return list(all_coins)


class ResourceManager:
    """Manages CPU and memory resources for optimization"""

    def __init__(self, max_cpu_percent=90, max_memory_percent=85, check_interval=5):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.paused = False
        self.last_check = 0

    def should_pause(self):
        """Check if processing should be paused due to high resource usage"""
        # Only check periodically to avoid overhead
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return self.paused

        self.last_check = current_time

        # Check CPU usage - use interval=None for immediate reading
        cpu_percent = psutil.cpu_percent(interval=None)

        # Check memory usage
        memory_percent = psutil.virtual_memory().percent

        # Determine if we should pause - only if extremely high
        if (
            cpu_percent > self.max_cpu_percent + 5
            or memory_percent > self.max_memory_percent + 5
        ):
            if not self.paused:
                logging.info(
                    f"Pausing due to high resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
                )
                self.paused = True
        else:
            if self.paused:
                logging.info(
                    f"Resuming processing: CPU {cpu_percent}%, Memory {memory_percent}%"
                )
                self.paused = False

        return self.paused

    async def wait_for_resources(self):
        """Wait until resource usage is below thresholds"""
        while self.should_pause():
            await asyncio.sleep(self.check_interval)


class GPUManager:
    """Manages GPU resources for optimization"""

    def __init__(
        self, gpu_type="auto", gpu_id=0, max_memory_percent=85, check_interval=5
    ):
        self.gpu_type = gpu_type
        self.gpu_id = gpu_id
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.paused = False
        self.last_check = 0
        self.available = False
        self.initialized = False

        # Initialize GPU detection
        self.detect_and_initialize()

    def detect_and_initialize(self):
        """Detect and initialize GPU"""
        if self.gpu_type == "auto":
            self.gpu_type = detect_gpu_type()

        if self.gpu_type == "none":
            logging.info("No GPU detected")
            self.available = False
            self.initialized = True
            return

        try:
            if self.gpu_type == "nvidia":
                self.initialize_nvidia()
            elif self.gpu_type == "amd":
                self.initialize_amd()
            else:
                logging.warning(f"Unknown GPU type: {self.gpu_type}")
                self.available = False
        except Exception as e:
            logging.error(f"Error initializing GPU: {e}")
            self.available = False

        self.initialized = True

    def initialize_nvidia(self):
        """Initialize NVIDIA GPU"""
        try:
            import cupy as cp

            # Set device
            cp.cuda.Device(self.gpu_id).use()

            # Get device properties
            device_props = cp.cuda.runtime.getDeviceProperties(self.gpu_id)

            # Log GPU info
            logging.info(f"NVIDIA GPU initialized: {device_props['name'].decode()}")
            logging.info(
                f"  Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB"
            )
            logging.info(
                f"  Compute Capability: {device_props['major']}.{device_props['minor']}"
            )

            self.available = True

        except Exception as e:
            logging.error(f"Failed to initialize NVIDIA GPU: {e}")
            self.available = False

    def initialize_amd(self):
        """Initialize AMD GPU"""
        try:
            import pyopencl as cl

            # Find AMD platform
            platforms = cl.get_platforms()
            amd_platform = None
            for platform in platforms:
                if "AMD" in platform.name:
                    amd_platform = platform
                    break

            if amd_platform is None:
                logging.error("No AMD platform found")
                self.available = False
                return

            # Get devices
            devices = amd_platform.get_devices()
            if self.gpu_id >= len(devices):
                logging.error(
                    f"GPU ID {self.gpu_id} out of range, max is {len(devices)-1}"
                )
                self.available = False
                return

            device = devices[self.gpu_id]

            # Log GPU info
            logging.info(f"AMD GPU initialized: {device.name}")
            logging.info(f"  Memory: {device.global_mem_size / (1024**3):.2f} GB")
            logging.info(f"  OpenCL Version: {device.version}")

            # Store context and queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)

            self.available = True

        except Exception as e:
            logging.error(f"Failed to initialize AMD GPU: {e}")
            self.available = False

    def get_memory_usage(self):
        """Get GPU memory usage percentage"""
        if not self.available:
            return 0

        try:
            if self.gpu_type == "nvidia":
                return self.get_nvidia_memory_usage()
            elif self.gpu_type == "amd":
                return self.get_amd_memory_usage()
        except Exception as e:
            logging.error(f"Error getting GPU memory usage: {e}")

        return 0

    def get_nvidia_memory_usage(self):
        """Get NVIDIA GPU memory usage percentage"""
        try:
            import cupy as cp

            # Get memory info
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            percent = (used / total) * 100

            return percent

        except Exception as e:
            logging.error(f"Error getting NVIDIA GPU memory usage: {e}")
            return 0

    def get_amd_memory_usage(self):
        """Get AMD GPU memory usage percentage"""
        # Unfortunately, PyOpenCL doesn't provide a direct way to get memory usage
        # This is a placeholder - in a real implementation, you might use a system
        # command or another library to get this information
        return 50  # Assume 50% usage as a default

    def should_pause(self):
        """Check if processing should be paused due to high GPU usage"""
        if not self.available:
            return False

        # Only check periodically to avoid overhead
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return self.paused

        self.last_check = current_time

        # Check GPU memory usage
        memory_percent = self.get_memory_usage()

        # Determine if we should pause - only if extremely high
        if memory_percent > self.max_memory_percent:
            if not self.paused:
                logging.info(f"Pausing due to high GPU memory usage: {memory_percent}%")
                self.paused = True
        else:
            if self.paused:
                logging.info(f"Resuming processing: GPU memory usage {memory_percent}%")
                self.paused = False

        return self.paused

    async def wait_for_resources(self):
        """Wait until GPU resources are available"""
        if not self.available:
            return

        while self.should_pause():
            await asyncio.sleep(self.check_interval)


def process_with_cpu_optimization(
    worker_id,
    individual,
    overrides_list,
    config,
    shared_memory_files,
    hlcvs_shapes,
    hlcvs_dtypes,
    btc_usd_shared_memory_files,
    btc_usd_dtypes,
    msss,
):
    """Process an individual with CPU optimization - must be at module level to be picklable"""
    # Try to optimize this process for CPU usage
    try:
        # Get current process
        process = psutil.Process()

        # Set CPU affinity if supported
        if hasattr(process, "cpu_affinity"):
            try:
                # Calculate which core this worker should use
                core_id = worker_id % psutil.cpu_count()
                process.cpu_affinity([core_id])
            except Exception as e:
                pass  # Ignore affinity errors

        # Set process priority
        if sys.platform == "win32":
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
        else:
            # Unix-like systems
            process.nice(0)  # Normal priority
    except Exception as e:
        pass  # Ignore process optimization errors

    # Now evaluate the individual
    return evaluate_individual_wrapper(
        individual,
        overrides_list,
        config,
        shared_memory_files,
        hlcvs_shapes,
        hlcvs_dtypes,
        btc_usd_shared_memory_files,
        btc_usd_dtypes,
        msss,
    )


class DistributedOptimizer:
    """Base class for both server and client implementations"""

    def __init__(self, args):
        self.args = args
        self.node_id = str(uuid.uuid4())[:8]
        self.context = zmq.asyncio.Context()

    async def run(self):
        raise NotImplementedError("Subclasses must implement run()")


class OptimizationServer(DistributedOptimizer):
    """Server that distributes optimization tasks and collects results"""

    def __init__(self, args):
        super().__init__(args)
        self.config_path = args.config
        self.port = args.port
        self.clients = {}
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.population = []
        self.pareto_front = []
        self.objectives_dict = {}
        self.index_to_entry = {}
        self.iteration = 0
        self.scoring_keys = None
        self.n_objectives = None
        self.param_bounds = None
        self.sig_digits = 6
        self.seen_hashes = {}
        self.task_progress = {}
        self.use_compression = args.compression

        # GPU-related parameters
        self.use_gpu = args.use_gpu
        self.gpu_id = args.gpu_id
        self.hybrid_mode = args.hybrid_mode
        self.gpu_batch_size = args.gpu_batch_size
        self.cpu_batch_size = args.cpu_batch_size
        self.use_compression = args.compression

        # Initialize GPU manager if requested
        if self.use_gpu:
            self.gpu_manager = GPUManager(gpu_id=self.gpu_id)
        else:
            self.gpu_manager = None

        # Create thread pool for I/O-bound operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # ZMQ sockets
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.pub_socket = self.context.socket(zmq.PUB)

        # Task generation parameters - increased batch size
        self.batch_size = args.batch_size
        self.population_size = 100  # Will be updated from config

        # Checkpointing
        self.checkpoint_interval = args.checkpoint_interval
        self.last_checkpoint = time.time()

    async def setup(self):
        """Load config and prepare for optimization"""
        logging.info(f"Loading config {self.config_path}")
        self.config = load_config(self.config_path)
        self.config = format_config(self.config)

        # Set up results directory
        self.setup_results_dir()

        # Extract optimization parameters
        self.population_size = self.config["optimize"]["population_size"]
        self.sig_digits = self.config["optimize"].get(
            "round_to_n_significant_digits", 6
        )
        self.param_bounds = self.config["optimize"]["bounds"]

        # Set up Pareto store
        self.store = ParetoStore(self.results_dir)

        # Bind sockets
        self.router_socket.bind(f"tcp://*:{self.port}")
        self.pub_socket.bind(f"tcp://*:{self.port+1}")

        logging.info(
            f"Server started on port {self.port} (router) and {self.port+1} (publisher)"
        )

        # Try to load checkpoint if it exists
        if self.args.resume and self.try_load_checkpoint():
            logging.info("Resumed from checkpoint")
        else:
            # Initialize population
            self.initialize_population()

    def setup_results_dir(self):
        """Set up results directory for storing optimization results"""
        exchanges = self.config["backtest"]["exchanges"]
        exchanges_fname = (
            "combined"
            if self.config["backtest"]["combine_ohlcvs"]
            else "_".join(exchanges)
        )
        date_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")

        # Get coins from config
        if "coins" in self.config["backtest"] and self.config["backtest"]["coins"]:
            coins = sorted(
                set([x for y in self.config["backtest"]["coins"].values() for x in y])
            )
        else:
            coins = ["unknown"]

        coins_fname = "_".join(coins) if len(coins) <= 6 else f"{len(coins)}_coins"
        hash_snippet = uuid.uuid4().hex[:8]

        n_days = int(
            round(
                (
                    date_to_ts(self.config["backtest"]["end_date"])
                    - date_to_ts(self.config["backtest"]["start_date"])
                )
                / (1000 * 60 * 60 * 24)
            )
        )

        self.results_dir = make_get_filepath(
            f"optimize_results/{date_fname}_{exchanges_fname}_{n_days}days_{coins_fname}_{hash_snippet}/"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        self.config["results_dir"] = self.results_dir

        # Save initial config
        with open(os.path.join(self.results_dir, "initial_config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        logging.info(f"Results will be saved to {self.results_dir}")

    def initialize_population(self):
        """Create initial population for optimization"""
        bounds = [(low, high) for param, (low, high) in self.param_bounds.items()]

        # Create random individuals
        for _ in range(self.population_size):
            individual = []
            for i, (low, high) in enumerate(bounds):
                if low == high:
                    individual.append(low)
                else:
                    individual.append(
                        round_floats(np.random.uniform(low, high), self.sig_digits)
                    )

            # Add to population
            self.population.append(individual)

        logging.info(f"Initialized population with {len(self.population)} individuals")

    def try_load_checkpoint(self):
        """Try to load optimization state from checkpoint"""
        checkpoint_path = os.path.join(self.results_dir, "checkpoint.json")

        if not os.path.exists(checkpoint_path):
            return False

        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)

            # Load state from checkpoint
            self.population = checkpoint.get("population", [])
            self.pareto_front = checkpoint.get("pareto_front", [])
            self.objectives_dict = {
                int(k): tuple(v)
                for k, v in checkpoint.get("objectives_dict", {}).items()
            }
            self.index_to_entry = {
                int(k): v for k, v in checkpoint.get("index_to_entry", {}).items()
            }
            self.iteration = checkpoint.get("iteration", 0)
            self.seen_hashes = checkpoint.get("seen_hashes", {})

            logging.info(
                f"Loaded checkpoint: iteration {self.iteration}, population size {len(self.population)}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return False

    def save_checkpoint(self):
        """Save current optimization state to checkpoint"""
        checkpoint_path = os.path.join(self.results_dir, "checkpoint.json")

        try:
            # Prepare checkpoint data
            checkpoint = {
                "timestamp": time.time(),
                "iteration": self.iteration,
                "population": self.population,
                "pareto_front": self.pareto_front,
                "objectives_dict": {
                    str(k): list(v) for k, v in self.objectives_dict.items()
                },
                "index_to_entry": {str(k): v for k, v in self.index_to_entry.items()},
                "seen_hashes": self.seen_hashes,
            }

            # Save to temporary file first to avoid corruption if interrupted
            temp_path = checkpoint_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(checkpoint, f)

            # Rename to final path
            os.replace(temp_path, checkpoint_path)

            logging.info(
                f"Saved checkpoint: iteration {self.iteration}, population size {len(self.population)}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            return False

    async def handle_client_message(self, client_id, message):
        """Process messages from clients"""
        # Decode client_id if it's bytes
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8")

        msg_type = message.get("type")

        # Handle special message types
        if msg_type == "not_ready":
            # Client is not ready yet, don't send tasks
            logging.info(f"Client {client_id} reported it's not ready yet")
            return

        # Special handling for registration messages
        if msg_type == "register":
            # New client registration
            hostname = message.get("hostname", "unknown")
            resource_info = message.get("resources", {})

            # If client already exists, update its info
            if client_id in self.clients:
                self.clients[client_id].update(
                    {
                        "hostname": hostname,
                        "resources": resource_info,
                        "last_seen": time.time(),
                    }
                )
                logging.info(f"Updated existing client: {hostname} ({client_id})")
            else:
                # New client
                self.clients[client_id] = {
                    "hostname": hostname,
                    "resources": resource_info,
                    "last_seen": time.time(),
                    "status": "idle",
                    "tasks_completed": 0,
                }
                logging.info(f"New client registered: {hostname} ({client_id})")

                # Log GPU information if available
                if "gpu" in resource_info:
                    gpu_info = resource_info["gpu"]
                    if gpu_info.get("available", False):
                        logging.info(
                            f"Client {client_id} has {gpu_info.get('type', 'unknown')} GPU: {gpu_info.get('name', 'unknown')}"
                        )

            # Send registration confirmation
            config_message = {"type": "registered", "config": self.config}
            await self.send_message_to_client(client_id, config_message)

            # Clear any re-registration requests for this client
            if (
                hasattr(self, "reregister_requests")
                and client_id in self.reregister_requests
            ):
                del self.reregister_requests[client_id]
            return

        # If client is not registered, request registration
        if client_id not in self.clients:
            # Use a class variable to track clients we've asked to re-register
            if not hasattr(self, "reregister_requests"):
                self.reregister_requests = {}

            # Only send re-registration request if we haven't sent one recently
            current_time = time.time()
            if (
                client_id not in self.reregister_requests
                or current_time - self.reregister_requests[client_id] > 30
            ):
                logging.info(
                    f"Client {client_id} reconnected, requesting re-registration"
                )
                await self.send_message_to_client(client_id, {"type": "reregister"})
                self.reregister_requests[client_id] = current_time
            return

        # Handle other message types
        if msg_type == "heartbeat":
            # Update client's last seen timestamp
            if client_id in self.clients:
                self.clients[client_id]["last_seen"] = time.time()
                # Update resource info if provided
                if "resources" in message:
                    self.clients[client_id]["resources"] = message["resources"]
                await self.send_message_to_client(client_id, {"type": "ack"})

        elif msg_type == "ready":
            # Client is ready for tasks
            if client_id in self.clients:
                self.clients[client_id]["status"] = "idle"
                logging.info(f"Client {client_id} ready for tasks")
                # Send tasks if available
                await self.send_tasks_to_client(client_id)

        elif msg_type == "result":
            # Process optimization results
            task_id = message.get("task_id")
            results = message.get("results", [])

            if task_id in self.pending_tasks:
                # Mark task as completed
                self.pending_tasks.pop(task_id)
                self.completed_tasks[task_id] = time.time()

                # Update client status
                if client_id in self.clients:
                    self.clients[client_id]["status"] = "idle"
                    self.clients[client_id]["tasks_completed"] += 1

                # Process results
                await self.process_results(results)

                # Send more tasks
                await self.send_tasks_to_client(client_id)

                # Check if we should save a checkpoint
                current_time = time.time()
                if current_time - self.last_checkpoint > self.checkpoint_interval:
                    self.save_checkpoint()
                    self.last_checkpoint = current_time

        elif msg_type == "progress":
            # Handle progress updates
            await self.handle_progress_update(client_id, message)

        elif msg_type == "error":
            # Handle client errors
            task_id = message.get("task_id")
            error = message.get("error", "Unknown error")

            if task_id in self.pending_tasks:
                logging.error(
                    f"Client {client_id} reported error on task {task_id}: {error}"
                )
                # Return task to queue
                individuals = self.pending_tasks.pop(task_id)
                self.population.extend(individuals)

                # Update client status
                if client_id in self.clients:
                    self.clients[client_id]["status"] = "idle"

                # Send more tasks
                await self.send_tasks_to_client(client_id)

    async def send_message_to_client(self, client_id, message):
        """Send a message to a client with optional compression"""
        try:
            if (
                self.use_compression and sys.getsizeof(str(message)) > 1024
            ):  # Only compress messages > 1KB
                # Compress the message
                message_json = json.dumps(message).encode()
                compressed_data = zlib.compress(message_json)

                # Add compression flag
                wrapper = {"compressed": True, "data": compressed_data}

                # Send compressed message
                await self.router_socket.send_multipart(
                    [
                        client_id.encode() if isinstance(client_id, str) else client_id,
                        json.dumps(wrapper).encode(),
                    ]
                )
            else:
                # Send uncompressed message
                await self.router_socket.send_multipart(
                    [
                        client_id.encode() if isinstance(client_id, str) else client_id,
                        json.dumps(message).encode(),
                    ]
                )
        except Exception as e:
            logging.error(f"Error sending message to client {client_id}: {e}")

    async def handle_progress_update(self, client_id, message):
        """Handle progress updates from clients"""
        task_id = message.get("task_id")
        processed = message.get("processed", 0)
        total = message.get("total", 0)
        valid_results = message.get("valid_results", 0)

        if task_id in self.pending_tasks:
            # Update task progress
            self.task_progress[task_id] = {
                "client_id": client_id,
                "processed": processed,
                "total": total,
                "valid_results": valid_results,
                "last_update": time.time(),
            }

            # Log progress
            logging.info(
                f"Task {task_id} progress: {processed}/{total} ({valid_results} valid)"
            )

            # Broadcast progress to all clients
            await self.pub_socket.send(
                json.dumps(
                    {
                        "type": "task_progress",
                        "task_id": task_id,
                        "client_id": client_id,
                        "processed": processed,
                        "total": total,
                        "valid_results": valid_results,
                    }
                ).encode()
            )

    async def send_tasks_to_client(self, client_id):
        """Send optimization tasks to a client with rate limiting to prevent overloading"""
        if not self.population:
            # No more tasks to send
            return

        # Ensure client_id is properly decoded
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8")

        # Check if client exists
        client_info = self.clients.get(client_id)
        if not client_info:
            logging.warning(f"Attempted to send task to unknown client {client_id}")
            return

        # Check if client is already busy
        if client_info.get("status") == "busy":
            logging.info(f"Client {client_id} is already busy, not sending more tasks")
            return

        # Check how many pending tasks this client already has
        client_pending_tasks = sum(
            1
            for task_id, task_info in self.pending_tasks.items()
            if isinstance(task_info, dict) and task_info.get("client_id") == client_id
        )

        # Get client resource information
        resource_info = client_info.get("resources", {})
        client_workers = resource_info.get("workers", 2)
        has_gpu = resource_info.get("gpu", {}).get("available", False)

        # Limit the number of pending tasks per client based on their worker count
        # GPU clients can handle more tasks
        max_pending_tasks = max(1, client_workers // (1 if has_gpu else 2))

        if client_pending_tasks >= max_pending_tasks:
            logging.info(
                f"Client {client_id} already has {client_pending_tasks} pending tasks, not sending more"
            )
            return

        # Determine batch size based on client resources
        client_cpu_count = resource_info.get("cpu_count", 4)

        # Scale batch size based on client's capabilities
        # GPU clients get larger batches
        if has_gpu:
            # GPU clients can handle larger batches
            adaptive_batch_size = min(
                max(
                    self.batch_size * 2,  # Double the base batch size for GPU clients
                    client_workers * 8,  # More individuals per worker for GPU
                ),
                len(self.population),
            )
        else:
            # CPU clients get standard batch sizes
            adaptive_batch_size = min(
                max(
                    self.batch_size,
                    client_workers * 3,  # At least 3 individuals per worker
                ),
                len(self.population),
            )

        # Get batch of individuals to evaluate
        individuals = self.population[:adaptive_batch_size]
        self.population = self.population[adaptive_batch_size:]

        # Create task
        task_id = str(uuid.uuid4())

        # Record task creation time and details
        self.pending_tasks[task_id] = {
            "individuals": individuals,
            "created_at": time.time(),
            "client_id": client_id,
        }

        # Initialize task progress
        self.task_progress[task_id] = {
            "client_id": client_id,
            "processed": 0,
            "total": len(individuals),
            "valid_results": 0,
            "last_update": time.time(),
        }

        # Send task to client
        task_message = {
            "type": "task",
            "task_id": task_id,
            "individuals": individuals,
            "param_bounds": self.param_bounds,
        }

        try:
            await self.send_message_to_client(client_id, task_message)

            # Update client status
            self.clients[client_id]["status"] = "busy"
            logging.info(
                f"Sent task {task_id} with {len(individuals)} individuals to client {client_id} "
                f"(workers: {client_workers}, GPU: {has_gpu})"
            )
        except Exception as e:
            logging.error(f"Error sending task to client {client_id}: {e}")

            # Return individuals to population
            self.population.extend(individuals)

            # Remove from pending tasks
            if task_id in self.pending_tasks:
                self.pending_tasks.pop(task_id)

            # Remove from task progress
            if task_id in self.task_progress:
                del self.task_progress[task_id]

    async def process_results(self, results):
        """Process optimization results from clients"""
        for result in results:
            individual = result.get("individual")
            config = result.get("config")
            analyses_combined = result.get("analyses_combined")

            # Skip invalid results
            if not individual or not config or not analyses_combined:
                logging.warning(f"Skipping invalid result: missing required fields")
                continue

            # Extract scores
            if self.scoring_keys is None and "optimize" in config:
                self.scoring_keys = config["optimize"]["scoring"]
                self.n_objectives = len(self.scoring_keys)

            # Get objective values
            keys = [k for k in analyses_combined if k.startswith("w_")]
            scores = [analyses_combined.get(k) for k in sorted(keys)]

            if any(s is None or not isinstance(s, (float, int)) for s in scores):
                continue

            scores = tuple(float(s) for s in scores)
            self.iteration += 1
            index = self.iteration

            # Store objectives and entry
            self.objectives_dict[index] = scores
            self.index_to_entry[index] = config

            # Check if dominated by existing Pareto front
            if any(
                dominates(self.objectives_dict[idx], scores)
                for idx in self.pareto_front
            ):
                continue

            # Remove dominated entries
            dominated = [
                idx
                for idx in self.pareto_front
                if dominates(scores, self.objectives_dict[idx])
            ]
            for idx in dominated:
                old_entry = self.index_to_entry[idx]
                self.store.remove_entry(
                    self.store.hash_entry(
                        round_floats(old_entry, sig_digits=self.store.sig_digits)
                    )
                )

            # Update Pareto front
            self.pareto_front = [
                idx
                for idx in self.pareto_front
                if not dominates(scores, self.objectives_dict[idx])
            ]
            self.pareto_front.append(index)
            self.store.add_entry(config)

            # Generate new individuals based on good results
            self.generate_new_individuals(individual, scores)

            # Log Pareto front update
            if self.n_objectives:
                line = "(min,max): "
                for i, sk in enumerate(self.scoring_keys):
                    min_val = min(
                        self.objectives_dict[idx][i] for idx in self.pareto_front
                    )
                    max_val = max(
                        self.objectives_dict[idx][i] for idx in self.pareto_front
                    )
                    line += f"{sk}: ({min_val:.5f},{max_val:.5f})"
                    if i < self.n_objectives - 1:
                        line += " | "

                logging.info(
                    f"Upd PF | Iter: {self.iteration} | n memb: {len(self.pareto_front)} | {line}"
                )

    def generate_new_individuals(self, good_individual, scores):
        """Generate new individuals based on good results"""
        # Create mutations of the good individual
        for _ in range(3):  # Generate 3 mutations
            new_individual = []
            for i, val in enumerate(good_individual):
                param_name = list(self.param_bounds.keys())[i]
                low, high = self.param_bounds[param_name]

                if low == high:
                    new_individual.append(low)
                    continue

                # Add some noise to the parameter
                mutation_strength = np.random.choice(
                    [0.01, 0.05, 0.1]
                )  # Different mutation strengths
                if val != 0:
                    # Proportional mutation
                    delta = val * mutation_strength * np.random.normal()
                    new_val = val + delta
                else:
                    # Absolute mutation when value is zero
                    range_size = high - low
                    delta = range_size * mutation_strength * np.random.normal()
                    new_val = val + delta
                    # Ensure within bounds and round
                new_val = max(low, min(high, new_val))
                new_val = round_floats(new_val, self.sig_digits)
                new_individual.append(new_val)

            # Add to population if not seen before
            individual_hash = calc_hash(new_individual)
            if individual_hash not in self.seen_hashes:
                self.population.append(new_individual)
                self.seen_hashes[individual_hash] = True

    async def check_clients(self):
        """Check client status and handle disconnections"""
        current_time = time.time()
        disconnected_clients = []

        for client_id, client_info in list(self.clients.items()):
            # Check if client hasn't sent a heartbeat in 10 minutes
            if current_time - client_info["last_seen"] > 600:
                logging.warning(
                    f"Client {client_id} ({client_info['hostname']}) appears to be disconnected"
                )
                disconnected_clients.append(client_id)

                # If client had pending tasks, return them to the queue
                tasks_to_return = []
                for task_id, individuals in list(self.pending_tasks.items()):
                    if (
                        task_id in self.task_progress
                        and self.task_progress[task_id]["client_id"] == client_id
                    ):
                        tasks_to_return.append((task_id, individuals))

                # Return tasks to queue
                for task_id, individuals in tasks_to_return:
                    logging.info(
                        f"Returning task {task_id} from disconnected client {client_id} to queue"
                    )
                    self.population.extend(individuals)
                    self.pending_tasks.pop(task_id)

                    # Remove from task progress
                    if task_id in self.task_progress:
                        del self.task_progress[task_id]

        # Remove disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                self.clients.pop(client_id)

    async def monitor_tasks(self):
        """Monitor tasks and restart any that appear stuck"""
        check_interval = 60  # Check every minute

        while True:
            current_time = time.time()
            stuck_tasks = []

            # Check for stuck tasks
            for task_id, individuals in list(self.pending_tasks.items()):
                # Get task progress if available
                task_progress = self.task_progress.get(task_id)

                if task_progress:
                    last_update = task_progress.get("last_update", 0)
                    client_id = task_progress.get("client_id")

                    # Check if task is stuck (no progress update in 10 minutes)
                    if current_time - last_update > 600:  # 10 minutes
                        logging.warning(
                            f"Task {task_id} from client {client_id} appears to be stuck - no updates for {(current_time - last_update) / 60:.1f} minutes"
                        )
                        stuck_tasks.append((task_id, individuals, client_id))
                else:
                    # No progress info - check if task is old (created more than 15 minutes ago)
                    # This is a fallback for tasks that never reported progress
                    task_created = self.pending_tasks.get(task_id, {}).get(
                        "created_at", current_time
                    )
                    if current_time - task_created > 900:  # 15 minutes
                        logging.warning(
                            f"Task {task_id} appears to be stuck - no progress reports received"
                        )
                        stuck_tasks.append((task_id, individuals, None))

            # Handle stuck tasks
            for task_id, individuals, client_id in stuck_tasks:
                logging.warning(f"Returning stuck task {task_id} to queue")

                # Return individuals to population
                self.population.extend(individuals)

                # Remove from pending tasks
                if task_id in self.pending_tasks:
                    self.pending_tasks.pop(task_id)

                # Remove from task progress
                if task_id in self.task_progress:
                    del self.task_progress[task_id]

                # Mark client as disconnected if we haven't heard from it in a while
                if client_id and client_id in self.clients:
                    client_last_seen = self.clients[client_id].get("last_seen", 0)
                    if current_time - client_last_seen > 300:  # 5 minutes
                        logging.warning(
                            f"Marking client {client_id} as disconnected due to stuck task"
                        )
                        self.clients.pop(client_id)

            # Wait for next check
            await asyncio.sleep(check_interval)

    async def broadcast_status(self):
        """Broadcast system status to all clients"""
        status = {
            "type": "status",
            "clients": len(self.clients),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "population_size": len(self.population),
            "pareto_front_size": len(self.pareto_front),
            "iteration": self.iteration,
        }

        await self.pub_socket.send(json.dumps(status).encode())

    async def run(self):
        """Main server loop"""
        await self.setup()

        # Start background tasks
        asyncio.create_task(self.status_loop())
        asyncio.create_task(self.monitor_tasks())

        # Main message handling loop
        while True:
            try:
                # Receive message from clients
                client_id, message_data = await self.router_socket.recv_multipart()

                # Parse message
                try:
                    message_json = message_data.decode()
                    message = json.loads(message_json)

                    # Check if message is compressed
                    if isinstance(message, dict) and message.get("compressed", False):
                        # Decompress the data
                        compressed_data = message.get("data")
                        if isinstance(compressed_data, str):
                            # Handle base64 encoded data if needed
                            import base64

                            compressed_data = base64.b64decode(compressed_data)

                        # Decompress
                        decompressed_data = zlib.decompress(compressed_data)
                        message = json.loads(decompressed_data.decode())
                except Exception as e:
                    logging.error(f"Error parsing message: {e}")
                    continue

                # Handle client message
                await self.handle_client_message(client_id, message)

            except Exception as e:
                logging.error(f"Error in server loop: {e}")
                import traceback

                traceback.print_exc()

    async def status_loop(self):
        """Background task to periodically check clients and broadcast status"""
        while True:
            try:
                await self.check_clients()
                await self.broadcast_status()

                # Check if we need to generate more individuals
                if (
                    len(self.population) < self.population_size // 2
                    and len(self.pareto_front) > 0
                ):
                    self.generate_more_individuals()

                # Save current state periodically
                self.save_state()

                # Check if we should save a checkpoint
                current_time = time.time()
                if current_time - self.last_checkpoint > self.checkpoint_interval:
                    self.save_checkpoint()
                    self.last_checkpoint = current_time

            except Exception as e:
                logging.error(f"Error in status loop: {e}")

            await asyncio.sleep(5)  # Run every 5 seconds

    def generate_more_individuals(self):
        """Generate more individuals when population is running low"""
        logging.info("Generating more individuals for the population")

        # Use existing Pareto front members as seeds
        pareto_individuals = []
        for idx in self.pareto_front:
            config = self.index_to_entry[idx]
            if "bot" in config:
                # Extract individual from config
                individual = []
                for pside in ["long", "short"]:
                    for key in sorted(config["bot"][pside]):
                        if key != "enforce_exposure_limit":
                            individual.append(config["bot"][pside][key])
                pareto_individuals.append(individual)

        # Generate new individuals through crossover and mutation
        new_individuals = []

        # Random individuals
        bounds = [(low, high) for param, (low, high) in self.param_bounds.items()]
        for _ in range(self.population_size // 4):
            individual = []
            for i, (low, high) in enumerate(bounds):
                if low == high:
                    individual.append(low)
                else:
                    individual.append(
                        round_floats(np.random.uniform(low, high), self.sig_digits)
                    )
            new_individuals.append(individual)

        # Mutations of Pareto front members
        if pareto_individuals:
            for _ in range(self.population_size // 4):
                # Select random Pareto individual
                base_individual = pareto_individuals[
                    np.random.randint(0, len(pareto_individuals))
                ]

                # Create mutation
                new_individual = []
                for i, val in enumerate(base_individual):
                    param_name = list(self.param_bounds.keys())[i]
                    low, high = self.param_bounds[param_name]

                    if low == high:
                        new_individual.append(low)
                        continue

                    # Add some noise to the parameter
                    mutation_strength = np.random.choice([0.01, 0.05, 0.1])
                    if val != 0:
                        delta = val * mutation_strength * np.random.normal()
                        new_val = val + delta
                    else:
                        range_size = high - low
                        delta = range_size * mutation_strength * np.random.normal()
                        new_val = val + delta

                    # Ensure within bounds and round
                    new_val = max(low, min(high, new_val))
                    new_val = round_floats(new_val, self.sig_digits)
                    new_individual.append(new_val)

                new_individuals.append(new_individual)

        # Crossovers between Pareto front members
        if len(pareto_individuals) >= 2:
            for _ in range(self.population_size // 4):
                # Select two random Pareto individuals
                idx1, idx2 = np.random.choice(len(pareto_individuals), 2, replace=False)
                parent1 = pareto_individuals[idx1]
                parent2 = pareto_individuals[idx2]

                # Create crossover
                child = []
                for i in range(len(parent1)):
                    # Uniform crossover with some probability
                    if np.random.random() < 0.5:
                        val = parent1[i]
                    else:
                        val = parent2[i]

                    # Small mutation
                    param_name = list(self.param_bounds.keys())[i]
                    low, high = self.param_bounds[param_name]

                    if low != high:
                        if val != 0:
                            delta = val * 0.01 * np.random.normal()
                            val = val + delta
                        else:
                            range_size = high - low
                            delta = range_size * 0.01 * np.random.normal()
                            val = val + delta

                        # Ensure within bounds and round
                        val = max(low, min(high, val))
                        val = round_floats(val, self.sig_digits)

                    child.append(val)

                new_individuals.append(child)

        # Add new individuals to population
        for individual in new_individuals:
            individual_hash = calc_hash(individual)
            if individual_hash not in self.seen_hashes:
                self.population.append(individual)
                self.seen_hashes[individual_hash] = True

        logging.info(f"Added {len(new_individuals)} new individuals to population")

    def save_state(self):
        """Save current optimization state"""
        # Save Pareto front
        pareto_path = os.path.join(self.results_dir, "pareto_front.json")
        pareto_data = []

        for idx in self.pareto_front:
            entry = {
                "config": self.index_to_entry[idx],
                "scores": self.objectives_dict[idx],
            }
            pareto_data.append(entry)

        with open(pareto_path, "w") as f:
            json.dump(pareto_data, f, indent=2)

        # Save server status
        status_path = os.path.join(self.results_dir, "server_status.json")
        status = {
            "timestamp": time.time(),
            "clients": len(self.clients),
            "client_details": self.clients,
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "population_size": len(self.population),
            "pareto_front_size": len(self.pareto_front),
            "iteration": self.iteration,
        }

        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)


class OptimizationClient(DistributedOptimizer):
    """Client that receives optimization tasks and returns results"""

    def __init__(self, args):
        super().__init__(args)
        self.server_address = args.server
        self.hostname = socket.gethostname()

        # Resource management - more aggressive defaults
        self.max_cpu_percent = args.max_cpu
        self.max_memory_percent = args.max_memory
        self.aggressiveness = args.aggressiveness
        self.optimize_memory_flag = args.optimize_memory
        self.use_compression = args.compression

        # GPU settings
        self.use_gpu = args.use_gpu
        self.gpu_type = args.gpu_type
        self.gpu_id = args.gpu_id
        self.max_gpu_memory_percent = args.max_gpu_memory

        # Initialize GPU manager if using GPU
        if self.use_gpu:
            self.gpu_manager = GPUManager(
                gpu_id=self.gpu_id,
            )
        else:
            self.gpu_manager = None

        # Adjust resource limits based on aggressiveness
        self.target_cpu_percent = max(
            30, self.max_cpu_percent - 20
        )  # Target slightly below max
        self.target_memory_percent = max(30, self.max_memory_percent - 20)

        self.resource_manager = ResourceManager(
            max_cpu_percent=self.max_cpu_percent,
            max_memory_percent=self.max_memory_percent,
        )

        # Set process priority
        self.set_process_priority()

        # Detect CPU features
        self.detect_cpu_features()

        # Set CPU affinity if supported
        self.set_cpu_affinity()

        # Start with more workers based on system
        total_cpus = multiprocessing.cpu_count()
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)

        # Log CPU information
        logging.info(
            f"System has {physical_cpus} physical cores, {logical_cpus} logical cores"
        )

        if args.workers > 0:
            self.n_workers = args.workers
        else:
            # More aggressive default worker count based on system type
            if sys.platform == "darwin":  # macOS - be a bit more conservative
                self.n_workers = max(1, int(physical_cpus * 0.7 * self.aggressiveness))
            elif (
                sys.platform == "win32" and logical_cpus > physical_cpus * 1.5
            ):  # Windows with hyperthreading
                # On systems with hyperthreading, we can use more workers
                self.n_workers = max(1, int(logical_cpus * 0.7 * self.aggressiveness))
            else:  # Linux or other systems
                self.n_workers = max(1, int(total_cpus * 0.7 * self.aggressiveness))

            # If using GPU, reduce CPU worker count slightly to avoid resource contention
        if self.use_gpu and self.gpu_manager and self.gpu_manager.available:
            self.n_workers = max(1, int(self.n_workers * 0.8))
            logging.info(f"Reduced worker count to {self.n_workers} due to GPU usage")

        logging.info(
            f"Starting with {self.n_workers} workers (out of {total_cpus} CPUs)"
        )

        # Initialize process pool with max workers
        # We'll adjust the actual number of workers used without recreating the pool
        self.max_workers = multiprocessing.cpu_count()
        self.n_workers = min(
            self.max_workers, max(1, int(self.max_workers * 0.7 * self.aggressiveness))
        )
        self.active_workers = (
            self.n_workers
        )  # Track how many workers we're actually using

        # Create process pool with max capacity
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        # Create a semaphore to limit concurrent tasks
        self.worker_semaphore = None  # Will be initialized in setup

        # Create process pool for CPU-bound tasks
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)

        # Create thread pool for I/O-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Prewarm the process pool
        self.prewarm_process_pool()

        # ZMQ sockets
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.sub_socket = self.context.socket(zmq.SUB)

        # Task processing
        self.current_task = None
        self.config = None
        self.evaluator = None
        self.shared_memory_files = {}
        self.hlcvs_dict = {}
        self.btc_usd_data_dict = {}
        self.hlcvs_shapes = {}
        self.hlcvs_dtypes = {}
        self.btc_usd_dtypes = {}
        self.btc_usd_shared_memory_files = {}
        self.msss = {}

        # Performance tracking
        self.last_cpu_readings = []
        self.last_mem_readings = []
        self.max_readings = 5  # Number of readings to keep for smoothing
        self.last_worker_adjustment = time.time()
        self.adjustment_cooldown = 5  # Seconds between worker count adjustments

        # Hybrid mode settings
        self.hybrid_mode = args.hybrid_mode
        self.gpu_batch_size = args.gpu_batch_size
        self.cpu_batch_size = args.cpu_batch_size

    # Add this method to the OptimizationClient class

    def ensure_worker_cpu_intensive(self, worker_id):
        """
        Ensure that worker processes are CPU-intensive by setting process affinity
        and priority appropriately.
        """
        try:
            # Get current process
            process = psutil.Process()

            # Set CPU affinity if supported
            if (
                hasattr(process, "cpu_affinity")
                and self.max_workers <= psutil.cpu_count()
            ):
                # Try to assign each worker to a specific CPU core
                # This can improve performance by reducing context switching
                try:
                    # Calculate which core this worker should use
                    core_id = worker_id % psutil.cpu_count()
                    process.cpu_affinity([core_id])
                    logging.debug(f"Worker {worker_id} assigned to CPU core {core_id}")
                except Exception as e:
                    logging.debug(f"Could not set CPU affinity: {e}")

            # Set process priority
            if sys.platform == "win32":
                if self.args.priority == "high":
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
                elif self.args.priority == "low":
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                # Unix-like systems
                if self.args.priority == "high":
                    try:
                        process.nice(-10)  # Higher priority (requires root)
                    except:
                        process.nice(0)  # Normal priority
                elif self.args.priority == "low":
                    process.nice(10)  # Lower priority
                else:
                    process.nice(0)  # Normal priority

            # On Linux, set I/O priority to be lower
            if sys.platform.startswith("linux"):
                try:
                    import os

                    os.system(f"ionice -c 2 -n 7 -p {process.pid}")
                except:
                    pass
            return True
        except Exception as e:
            logging.error(f"Error setting worker process properties: {e}")
            return False

    def set_process_priority(self):
        """
        Set process priority based on the client's configuration.
        """
        try:
            # Get current process
            p = psutil.Process(os.getpid())

            if self.args.priority == "low":
                # Set to below normal priority
                if sys.platform == "win32":
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(10)  # Lower priority on Unix
                logging.info("Set process priority to low")

            elif self.args.priority == "normal":
                # Set to normal priority
                if sys.platform == "win32":
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(0)  # Normal priority on Unix
                logging.info("Set process priority to normal")

            elif self.args.priority == "high":
                # Set to above normal priority
                if sys.platform == "win32":
                    p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(-10)  # Higher priority on Unix (requires root)
                logging.info("Set process priority to high")

        except Exception as e:
            logging.warning(f"Could not set process priority: {e}")

    def set_cpu_affinity(self):
        """
        Set CPU affinity to distribute processes across all available cores.
        This can improve performance by preventing the OS from moving processes
        between cores too frequently.
        """
        try:
            # Only available on some platforms
            if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
                # Get current process
                pid = os.getpid()

                # Get available CPUs
                available_cpus = list(os.sched_getaffinity(pid))

                # Log current affinity
                logging.info(f"Current CPU affinity: {available_cpus}")

                # We'll keep the current affinity as it's likely already optimal
                # But we log it for debugging purposes
        except Exception as e:
            logging.debug(f"Could not get/set CPU affinity: {e}")

    def detect_cpu_features(self):
        """
        Detect CPU features that could be used for optimization.
        """
        try:
            import platform
            import subprocess

            cpu_info = {}

            # Different approaches based on platform
            if platform.system() == "Linux":
                # Use lscpu on Linux
                try:
                    output = subprocess.check_output("lscpu", shell=True).decode()
                    for line in output.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            cpu_info[key.strip()] = value.strip()
                except:
                    pass

            elif platform.system() == "Darwin":  # macOS
                # Use sysctl on macOS
                try:
                    output = subprocess.check_output(
                        "sysctl -a | grep machdep.cpu", shell=True
                    ).decode()
                    for line in output.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            cpu_info[key.strip()] = value.strip()
                except:
                    pass

            elif platform.system() == "Windows":
                # Use wmic on Windows
                try:
                    output = subprocess.check_output(
                        "wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors /format:list",
                        shell=True,
                    ).decode()
                    for line in output.split("\n"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            cpu_info[key.strip()] = value.strip()
                except:
                    pass

            # Log CPU information
            if cpu_info:
                logging.info(f"CPU information: {cpu_info}")

                # Check for specific optimizations
                avx2_available = False
                if platform.system() == "Linux":
                    avx2_available = "avx2" in cpu_info.get("Flags", "").lower()
                elif platform.system() == "Darwin":
                    avx2_available = (
                        "avx2" in cpu_info.get("machdep.cpu.features", "").lower()
                    )

                if avx2_available:
                    logging.info(
                        "AVX2 instructions available - Rust code should use these automatically"
                    )

        except Exception as e:
            logging.debug(f"Could not detect CPU features: {e}")

    def prewarm_process_pool(self):
        """
        Prewarm the process pool by submitting dummy tasks.
        This creates the worker processes upfront so they're ready when real tasks arrive.
        """
        logging.info(f"Prewarming process pool with {self.max_workers} workers...")

        # Submit one task per worker
        futures = []
        for i in range(self.max_workers):
            futures.append(self.process_pool.submit(dummy_task, i))

        # Wait for all tasks to complete
        for future in futures:
            future.result()

        logging.info("Process pool prewarming complete")

    def optimize_memory_usage(self):
        """
        Optimize memory usage by clearing unnecessary caches and
        triggering garbage collection when memory usage is high.
        """
        if not self.optimize_memory_flag:
            return False

        import gc

        # Get current memory usage
        memory_percent = psutil.virtual_memory().percent

        # If memory usage is high, take action
        if memory_percent > self.max_memory_percent - 10:
            logging.info(
                f"Memory usage is high ({memory_percent}%) - optimizing memory"
            )

            # Force garbage collection
            gc.collect()

            # Clear any caches if they exist
            if hasattr(self.evaluator, "cache"):
                self.evaluator.cache.clear()

            # Log memory after optimization
            new_memory_percent = psutil.virtual_memory().percent
            logging.info(f"Memory usage after optimization: {new_memory_percent}%")

            return True

        return False

    async def reconnect_to_server(self):
        """Reconnect to server after a disconnection with better error handling"""
        logging.info("Attempting to reconnect to server...")

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                # Close existing sockets
                self.dealer_socket.close()
                self.sub_socket.close()

                # Wait a moment for sockets to close properly
                await asyncio.sleep(1)

                # Create new sockets
                self.dealer_socket = self.context.socket(zmq.DEALER)
                self.sub_socket = self.context.socket(zmq.SUB)

                # Set client ID in socket
                self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())

                # Set socket options for more reliable connections
                self.dealer_socket.setsockopt(
                    zmq.RECONNECT_IVL, 1000
                )  # Reconnect after 1 second
                self.dealer_socket.setsockopt(
                    zmq.RECONNECT_IVL_MAX, 10000
                )  # Max 10 seconds between reconnects
                self.dealer_socket.setsockopt(
                    zmq.LINGER, 0
                )  # Don't wait for unsent messages when closing
                self.dealer_socket.setsockopt(
                    zmq.TCP_KEEPALIVE, 1
                )  # Enable TCP keepalive
                self.dealer_socket.setsockopt(
                    zmq.TCP_KEEPALIVE_IDLE, 60
                )  # Seconds before sending keepalive probes
                self.dealer_socket.setsockopt(
                    zmq.TCP_KEEPALIVE_INTVL, 10
                )  # Interval between keepalive probes

                # Connect to server
                server_host, server_port = self.server_address.split(":")
                self.dealer_socket.connect(f"tcp://{server_host}:{server_port}")
                self.sub_socket.connect(f"tcp://{server_host}:{int(server_port)+1}")
                self.sub_socket.setsockopt(
                    zmq.SUBSCRIBE, b""
                )  # Subscribe to all messages

                # Give ZMQ connections time to establish
                await asyncio.sleep(2)

                # Register with server
                logging.info(f"Reconnection attempt {attempt}/{max_attempts}...")
                success = await self.register_with_server()

                if success:
                    logging.info("Successfully reconnected to server")
                    return True
                else:
                    logging.warning(f"Reconnection attempt {attempt} failed")

                    # Exponential backoff
                    await asyncio.sleep(min(30, 2**attempt))

            except asyncio.CancelledError:
                logging.warning("Reconnection attempt was cancelled")
                return False
            except Exception as e:
                logging.error(f"Error during reconnection attempt {attempt}: {e}")
                import traceback

                traceback.print_exc()

                # Exponential backoff
                await asyncio.sleep(min(30, 2**attempt))

        logging.error(f"Failed to reconnect after {max_attempts} attempts")
        return False

    async def setup(self):
        """Connect to server and register"""
        # Set client ID in socket
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())

        # Connect to server
        server_host, server_port = self.server_address.split(":")
        self.dealer_socket.connect(f"tcp://{server_host}:{server_port}")
        self.sub_socket.connect(f"tcp://{server_host}:{int(server_port)+1}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.worker_semaphore = asyncio.Semaphore(self.n_workers)

        # Prewarm the process pool (only once)
        self.prewarm_process_pool()

        # Give ZMQ connections time to establish
        await asyncio.sleep(1)

        # Register with server
        success = await self.register_with_server()

        if not success:
            logging.error("Failed to register with server, exiting")
            sys.exit(1)

    async def initialize_evaluator(self):
        """Initialize the optimization evaluator"""

        # Prepare data for each exchange
        self.hlcvs_dict = {}
        self.shared_memory_files = {}
        self.hlcvs_shapes = {}
        self.hlcvs_dtypes = {}
        self.msss = {}

        # Store per-exchange BTC arrays
        self.btc_usd_data_dict = {}
        self.btc_usd_shared_memory_files = {}
        self.btc_usd_dtypes = {}

        # Ensure the config has a coins section
        if "coins" not in self.config["backtest"]:
            self.config["backtest"]["coins"] = {}

        # Debug: Print current config state
        logging.info(
            f"Current config exchanges: {self.config['backtest']['exchanges']}"
        )
        logging.info(f"Current approved coins: {self.config['live']['approved_coins']}")

        # Try to get all eligible coins directly
        try:

            # First approach: Use the add_all_eligible_coins_to_config function
            await add_all_eligible_coins_to_config(self.config)
            logging.info(
                f"After add_all_eligible_coins_to_config: {self.config['live']['approved_coins']}"
            )

            # If that didn't work, try a direct approach
            if not any(self.config["live"]["approved_coins"].values()):
                all_coins = await get_all_eligible_coins(
                    self.config["backtest"]["exchanges"]
                )
                logging.info(
                    f"Direct approach found {len(all_coins)} coins: {all_coins[:10]}..."
                )

                if all_coins:
                    self.config["live"]["approved_coins"] = {
                        "long": all_coins,
                        "short": all_coins,
                    }
                    logging.info(f"Updated approved_coins with direct approach")
        except Exception as e:
            logging.error(f"Error getting eligible coins: {e}")
            import traceback

            traceback.print_exc()

        # Manually specify some common coins if we still don't have any
        if not any(self.config["live"]["approved_coins"].values()):
            logging.warning(
                "No coins found automatically, using fallback list of common coins"
            )
            common_coins = [
                "BTC",
                "ETH",
                "SOL",
                "XRP",
                "ADA",
                "DOGE",
                "MATIC",
                "DOT",
                "LINK",
                "AVAX",
            ]
            self.config["live"]["approved_coins"] = {
                "long": common_coins,
                "short": common_coins,
            }

        # Now prepare the backtest coins based on approved coins
        if not self.config["backtest"].get("coins"):
            self.config["backtest"]["coins"] = {}

        for exchange in self.config["backtest"]["exchanges"]:
            if exchange not in self.config["backtest"]["coins"]:
                self.config["backtest"]["coins"][exchange] = []

            # Add approved coins to backtest coins if not already there
            for side in ["long", "short"]:
                for coin in self.config["live"]["approved_coins"].get(side, []):
                    if coin not in self.config["backtest"]["coins"][exchange]:
                        self.config["backtest"]["coins"][exchange].append(coin)

        # Debug: Print final coin configuration
        for exchange in self.config["backtest"]["exchanges"]:
            logging.info(
                f"Exchange {exchange} coins: {self.config['backtest']['coins'].get(exchange, [])}"
            )

        if not any(self.config["backtest"]["coins"].values()):
            logging.error("No eligible coins found in configuration after all attempts")
            raise ValueError(
                "No eligible coins found. Please check your configuration."
            )

        if self.config["backtest"]["combine_ohlcvs"]:
            exchange = "combined"
            try:
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = (
                    await prepare_hlcvs_mss(self.config, exchange)
                )
                self.config["backtest"]["coins"][exchange] = coins
                self.hlcvs_dict[exchange] = hlcvs
                self.hlcvs_shapes[exchange] = hlcvs.shape
                self.hlcvs_dtypes[exchange] = hlcvs.dtype
                self.msss[exchange] = mss

                required_space = hlcvs.nbytes * 1.1  # Add 10% buffer
                check_disk_space(tempfile.gettempdir(), required_space)

                logging.info(f"Creating shared memory file for {exchange}...")
                validate_array(hlcvs, "hlcvs")
                shared_memory_file = create_shared_memory_file(hlcvs)
                self.shared_memory_files[exchange] = shared_memory_file

                if self.config["backtest"].get("use_btc_collateral", False):
                    self.btc_usd_data_dict[exchange] = btc_usd_prices
                else:
                    self.btc_usd_data_dict[exchange] = np.ones(
                        hlcvs.shape[0], dtype=np.float64
                    )

                validate_array(
                    self.btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}"
                )
                self.btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                    self.btc_usd_data_dict[exchange]
                )
                self.btc_usd_dtypes[exchange] = self.btc_usd_data_dict[exchange].dtype

                # If GPU is available, transfer data to GPU memory
                if self.use_gpu and self.gpu_manager and self.gpu_manager.available:
                    self.transfer_data_to_gpu(
                        exchange, hlcvs, self.btc_usd_data_dict[exchange]
                    )

            except Exception as e:
                logging.error(f"Error preparing combined data: {e}")
                import traceback

                traceback.print_exc()

        else:
            tasks = {}
            for exchange in self.config["backtest"]["exchanges"]:
                tasks[exchange] = asyncio.create_task(
                    prepare_hlcvs_mss(self.config, exchange)
                )

            for exchange in self.config["backtest"]["exchanges"]:
                try:
                    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = (
                        await tasks[exchange]
                    )

                    if not coins:
                        logging.warning(f"No coins found for exchange {exchange}")
                        continue

                    self.config["backtest"]["coins"][exchange] = coins
                    self.hlcvs_dict[exchange] = hlcvs
                    self.hlcvs_shapes[exchange] = hlcvs.shape
                    self.hlcvs_dtypes[exchange] = hlcvs.dtype
                    self.msss[exchange] = mss

                    required_space = hlcvs.nbytes * 1.1
                    check_disk_space(tempfile.gettempdir(), required_space)

                    logging.info(f"Creating shared memory file for {exchange}...")
                    validate_array(hlcvs, "hlcvs")
                    shared_memory_file = create_shared_memory_file(hlcvs)
                    self.shared_memory_files[exchange] = shared_memory_file

                    if self.config["backtest"].get("use_btc_collateral", False):
                        self.btc_usd_data_dict[exchange] = btc_usd_prices
                    else:
                        self.btc_usd_data_dict[exchange] = np.ones(
                            hlcvs.shape[0], dtype=np.float64
                        )

                    validate_array(
                        self.btc_usd_data_dict[exchange], f"btc_usd_data for {exchange}"
                    )
                    self.btc_usd_shared_memory_files[exchange] = (
                        create_shared_memory_file(self.btc_usd_data_dict[exchange])
                    )
                    self.btc_usd_dtypes[exchange] = self.btc_usd_data_dict[
                        exchange
                    ].dtype

                    # If GPU is available, transfer data to GPU memory
                    if self.use_gpu and self.gpu_manager and self.gpu_manager.available:
                        self.transfer_data_to_gpu(
                            exchange, hlcvs, self.btc_usd_data_dict[exchange]
                        )

                except Exception as e:
                    logging.error(f"Error preparing data for {exchange}: {e}")
                    import traceback

                    traceback.print_exc()

        # Check if we have any valid exchanges with data
        if not self.shared_memory_files:
            raise ValueError(
                "No valid exchanges with coin data found. Please check your configuration."
            )

        logging.info("Data preparation complete")

    def transfer_data_to_gpu(self, exchange, hlcvs, btc_usd_data):
        """Transfer market data to GPU memory if GPU is available"""
        if not self.use_gpu or not self.gpu_manager or not self.gpu_manager.available:
            return

        try:
            if self.gpu_manager.gpu_type == "nvidia":
                self.transfer_data_to_nvidia_gpu(exchange, hlcvs, btc_usd_data)
            elif self.gpu_manager.gpu_type == "amd":
                self.transfer_data_to_amd_gpu(exchange, hlcvs, btc_usd_data)
        except Exception as e:
            logging.error(f"Error transferring data to GPU: {e}")
            import traceback

            traceback.print_exc()

    def transfer_data_to_nvidia_gpu(self, exchange, hlcvs, btc_usd_data):
        """Transfer data to NVIDIA GPU using CuPy"""
        try:
            import cupy as cp

            # Transfer HLCV data to GPU
            logging.info(f"Transferring {exchange} HLCV data to NVIDIA GPU...")
            gpu_hlcvs = cp.array(hlcvs)

            # Transfer BTC/USD data to GPU
            gpu_btc_usd = cp.array(btc_usd_data)

            # Store GPU arrays
            if not hasattr(self, "gpu_data"):
                self.gpu_data = {}

            if exchange not in self.gpu_data:
                self.gpu_data[exchange] = {}

            self.gpu_data[exchange]["hlcvs"] = gpu_hlcvs
            self.gpu_data[exchange]["btc_usd"] = gpu_btc_usd

            logging.info(f"Successfully transferred {exchange} data to NVIDIA GPU")

            # Log memory usage
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            percent = (used / total) * 100
            logging.info(
                f"GPU memory usage: {used / (1024**2):.1f} MB / {total / (1024**2):.1f} MB ({percent:.1f}%)"
            )

        except Exception as e:
            logging.error(f"Failed to transfer data to NVIDIA GPU: {e}")

    def transfer_data_to_amd_gpu(self, exchange, hlcvs, btc_usd_data):
        """Transfer data to AMD GPU using PyOpenCL with better error handling"""
        try:
            import pyopencl as cl
            import pyopencl.array

            # Get context and queue from GPU manager
            ctx = self.gpu_manager.context
            queue = self.gpu_manager.queue

            # Check data sizes
            hlcvs_size_mb = hlcvs.nbytes / (1024 * 1024)
            btc_usd_size_mb = btc_usd_data.nbytes / (1024 * 1024)
            total_size_mb = hlcvs_size_mb + btc_usd_size_mb

            # Get available GPU memory
            device_memory_mb = self.gpu_manager.memory_total / (1024 * 1024)

            logging.info(f"Transferring {exchange} data to AMD GPU:")
            logging.info(f"  HLCV data size: {hlcvs_size_mb:.2f} MB")
            logging.info(f"  BTC/USD data size: {btc_usd_size_mb:.2f} MB")
            logging.info(f"  Total data size: {total_size_mb:.2f} MB")
            logging.info(f"  GPU memory: {device_memory_mb:.2f} MB")

            # Check if data is too large (leave 20% buffer for other GPU operations)
            max_allowed_mb = device_memory_mb * 0.8
            if total_size_mb > max_allowed_mb:
                logging.warning(
                    f"Data size ({total_size_mb:.2f} MB) exceeds 80% of GPU memory ({max_allowed_mb:.2f} MB). "
                    f"Skipping GPU transfer to avoid out-of-memory errors."
                )
                return False

            # Check for zero-sized arrays
            if hlcvs.size == 0 or btc_usd_data.size == 0:
                logging.warning(f"Cannot transfer empty arrays to GPU. Skipping.")
                return False

            # Transfer HLCV data to GPU
            logging.info(f"Transferring {exchange} HLCV data to AMD GPU...")
            gpu_hlcvs = cl.array.to_device(queue, hlcvs)

            # Transfer BTC/USD data to GPU
            gpu_btc_usd = cl.array.to_device(queue, btc_usd_data)

            # Store GPU arrays
            if not hasattr(self, "gpu_data"):
                self.gpu_data = {}

            if exchange not in self.gpu_data:
                self.gpu_data[exchange] = {}

            self.gpu_data[exchange]["hlcvs"] = gpu_hlcvs
            self.gpu_data[exchange]["btc_usd"] = gpu_btc_usd

            logging.info(f"Successfully transferred {exchange} data to AMD GPU")
            return True

        except cl.LogicError as e:
            if "INVALID_BUFFER_SIZE" in str(e):
                logging.error(f"GPU buffer size error: {e}")
                logging.error(
                    f"HLCV shape: {hlcvs.shape}, dtype: {hlcvs.dtype}, size: {hlcvs.nbytes / (1024*1024):.2f} MB"
                )
                logging.error(
                    f"BTC/USD shape: {btc_usd_data.shape}, dtype: {btc_usd_data.dtype}, size: {btc_usd_data.nbytes / (1024*1024):.2f} MB"
                )
                logging.error(
                    "The data is too large for the GPU memory. Try using --hybrid-mode with a smaller --gpu-batch-size."
                )
                return False
            else:
                logging.error(f"OpenCL error: {e}")
                return False
        except Exception as e:
            logging.error(f"Failed to transfer data to AMD GPU: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def process_task(self, task):
        """Process an optimization task with robust error handling and progress display"""
        task_id = task["task_id"]
        individuals = task["individuals"]
        overrides_list = self.config.get("optimize", {}).get("enable_overrides", [])

        # Show task information
        start_time = time.time()
        print(f"\n{'='*80}")
        print(
            f"Starting task {task_id} with {len(individuals)} individuals at {datetime.datetime.now().strftime('%H:%M:%S')}"
        )
        print(f"{'='*80}")

        # Determine processing mode based on available resources
        use_gpu = (
            self.use_gpu
            and self.gpu_manager
            and self.gpu_manager.available
            and hasattr(self, "gpu_data")
            and self.gpu_data
        )

        if use_gpu and self.hybrid_mode:
            print(f"Using hybrid CPU/GPU mode with {self.n_workers} CPU workers")
        elif use_gpu:
            print(f"Using GPU mode")
        else:
            print(f"Using CPU mode with {self.n_workers} workers")

        # Set up progress tracking
        total_individuals = len(individuals)
        processed = 0
        valid_results = 0
        results = []

        try:
            # Keep connection alive during task processing
            keep_alive_task = asyncio.create_task(self.keep_connection_alive(task_id))

            # Process individuals with resource management and worker limiting
            if use_gpu and not self.hybrid_mode:
                # GPU-only processing
                print("Processing with GPU...")
                results = await self.process_individuals_gpu(
                    individuals, overrides_list, task_id
                )
                processed = len(individuals)
                valid_results = len(results)
            elif use_gpu and self.hybrid_mode:
                # Hybrid processing - split between GPU and CPU
                gpu_count = min(len(individuals), self.gpu_batch_size)
                cpu_count = len(individuals) - gpu_count

                print(
                    f"Hybrid processing: {gpu_count} individuals on GPU, {cpu_count} on CPU"
                )

                # Process GPU batch
                if gpu_count > 0:
                    print("Starting GPU processing...")
                    gpu_individuals = individuals[:gpu_count]
                    gpu_results_task = asyncio.create_task(
                        self.process_individuals_gpu(
                            gpu_individuals, overrides_list, task_id
                        )
                    )

                # Process CPU batch
                if cpu_count > 0:
                    print("Starting CPU processing...")
                    cpu_individuals = individuals[gpu_count:]
                    cpu_results_task = asyncio.create_task(
                        self.process_individuals_cpu(
                            cpu_individuals, overrides_list, task_id
                        )
                    )

                # Gather results
                if gpu_count > 0:
                    gpu_results = await gpu_results_task
                    results.extend(gpu_results)
                    valid_results += len(gpu_results)
                    processed += gpu_count
                    print(
                        f"\nGPU processing complete: {len(gpu_results)} valid results from {gpu_count} individuals"
                    )

                if cpu_count > 0:
                    cpu_results = await cpu_results_task
                    results.extend(cpu_results)
                    valid_results += len(cpu_results)
                    processed += cpu_count
                    print(
                        f"\nCPU processing complete: {len(cpu_results)} valid results from {cpu_count} individuals"
                    )
            else:
                # CPU-only processing
                print("Processing with CPU...")
                results = await self.process_individuals_cpu(
                    individuals, overrides_list, task_id
                )
                processed = len(individuals)
                valid_results = len(results)

            # Cancel keep-alive task
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Calculate performance metrics
            if valid_results > 0:
                # Extract some key metrics for display
                best_adg = 0
                best_drawdown = 1.0
                best_sharpe = 0

                for result in results:
                    if "analyses_combined" in result:
                        metrics = result["analyses_combined"]
                        adg = metrics.get("w_adg", 0)
                        drawdown = metrics.get("w_drawdown_worst", 1.0)
                        sharpe = metrics.get("w_sharpe_ratio", 0)

                        best_adg = max(best_adg, adg)
                        best_drawdown = min(best_drawdown, drawdown)
                        best_sharpe = max(best_sharpe, sharpe)

                # Display best metrics
                print(f"Best metrics found:")
                print(f"  ADG: {best_adg:.6f}")
                print(f"  Drawdown: {best_drawdown:.6f}")
                print(f"  Sharpe: {best_sharpe:.6f}")

            # Send final results
            print(f"\n{'='*80}")
            print(
                f"Task {task_id} completed: {valid_results}/{total_individuals} valid results"
            )
            print(
                f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            )
            print(
                f"Processing speed: {total_individuals/elapsed_time:.2f} individuals/second"
            )
            print(f"{'='*80}")

            # Compress results if they're large
            if self.use_compression and len(results) > 10:
                print("Compressing results for transmission...")
                # Prepare message
                result_message = {
                    "type": "result",
                    "task_id": task_id,
                    "results": results,
                }

                # Convert to JSON and compress
                result_json = json.dumps(result_message).encode()
                compressed_data = zlib.compress(result_json)

                # Create wrapper with compressed data
                wrapper = {"compressed": True, "data": compressed_data}

                # Send compressed message
                await self.dealer_socket.send(json.dumps(wrapper).encode())
            else:
                # Send uncompressed results
                print("Sending results to server...")
                await self.dealer_socket.send(
                    json.dumps(
                        {"type": "result", "task_id": task_id, "results": results}
                    ).encode()
                )

            # Signal ready for more tasks
            await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())
            print("Ready for next task")

        except Exception as e:
            logging.error(f"Error processing task {task_id}: {e}")
            import traceback

            traceback.print_exc()

            # Report error to server
            await self.dealer_socket.send(
                json.dumps(
                    {"type": "error", "task_id": task_id, "error": str(e)}
                ).encode()
            )

            # Try to signal ready for more tasks
            try:
                await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())
            except:
                # If we can't send ready message, try to reconnect
                await self.reconnect_to_server()

    # Add a new method to keep the connection alive during long-running tasks
    async def keep_connection_alive(self, task_id):
        """Send periodic heartbeats during long-running tasks to prevent disconnection"""
        interval = 30  # Send a heartbeat every 30 seconds

        try:
            while True:
                # Send a heartbeat to keep the connection alive
                try:
                    resource_info = {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": psutil.virtual_memory().percent,
                        "workers": self.n_workers,
                    }

                    await self.dealer_socket.send(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "resources": resource_info,
                                "task_id": task_id,
                                "processing": True,
                            }
                        ).encode()
                    )

                    # Don't wait for response to avoid blocking
                except Exception as e:
                    logging.warning(f"Error sending keep-alive heartbeat: {e}")

                    # Try to reconnect if we can't send heartbeats
                    if isinstance(e, (zmq.ZMQError, ConnectionError)):
                        logging.error(
                            "Connection issue detected, attempting to reconnect"
                        )
                        await self.reconnect_to_server()

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            # This is expected when the task completes
            pass
        except Exception as e:
            logging.error(f"Error in keep_connection_alive: {e}")

    async def process_individuals_cpu(self, individuals, overrides_list, task_id):
        """Process individuals using CPU with better worker management and progress display"""
        results = []
        processed = 0
        valid_results = 0
        total = len(individuals)
        last_progress_update = time.time()
        start_time = time.time()

        # Create tasks for each individual
        tasks = []
        for i, individual in enumerate(individuals):
            # Create a task for each individual with a worker ID
            worker_id = i % self.n_workers  # Assign a worker ID based on position
            tasks.append(
                self.process_individual(
                    individual, overrides_list, self.worker_semaphore, worker_id
                )
            )

        # Create a progress bar with ETA
        with tqdm.tqdm(
            total=total,
            desc="Processing",
            unit="ind",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            # Process tasks as they complete
            for future in asyncio.as_completed(tasks):
                result = await future

                if result:
                    results.append(result)
                    valid_results += 1

                    # Extract some key metrics for display
                    if "analyses_combined" in result:
                        metrics = result["analyses_combined"]
                        adg = metrics.get("w_adg", 0)
                        drawdown = metrics.get("w_drawdown_worst", 0)
                        sharpe = metrics.get("w_sharpe_ratio", 0)

                        # Update progress bar with metrics
                        pbar.set_postfix(
                            valid=valid_results,
                            adg=f"{adg:.4f}",
                            dd=f"{drawdown:.4f}",
                            sharpe=f"{sharpe:.2f}",
                        )
                    else:
                        pbar.set_postfix(valid=valid_results)

                processed += 1
                pbar.update(1)

                # Send progress updates periodically
                current_time = time.time()
                if (
                    current_time - last_progress_update > 5
                ):  # Every 5 seconds (more frequent)
                    # Calculate speed and ETA
                    elapsed = current_time - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    remaining = (total - processed) / speed if speed > 0 else 0

                    # Format ETA
                    eta_hours, remainder = divmod(remaining, 3600)
                    eta_minutes, eta_seconds = divmod(remainder, 60)
                    eta_str = f"{int(eta_hours):02d}:{int(eta_minutes):02d}:{int(eta_seconds):02d}"

                    # Send progress to server
                    await self.dealer_socket.send(
                        json.dumps(
                            {
                                "type": "progress",
                                "task_id": task_id,
                                "processed": processed,
                                "total": total,
                                "valid_results": valid_results,
                                "speed": speed,
                                "eta": eta_str,
                            }
                        ).encode()
                    )
                    last_progress_update = current_time

                    # Also optimize memory if needed
                    self.optimize_memory_usage()

                    # Send a heartbeat to keep connection alive during long-running tasks
                    try:
                        await self.dealer_socket.send(
                            json.dumps({"type": "heartbeat"}).encode()
                        )
                    except Exception as e:
                        logging.debug(f"Error sending heartbeat during processing: {e}")

        # Final progress update
        await self.dealer_socket.send(
            json.dumps(
                {
                    "type": "progress",
                    "task_id": task_id,
                    "processed": processed,
                    "total": total,
                    "valid_results": valid_results,
                    "complete": True,
                }
            ).encode()
        )

        return results

    # Add this method to the OptimizationClient class

    async def monitor_cpu_utilization(self):
        """
        Monitor CPU utilization and adjust worker count more aggressively
        to ensure we're maximizing CPU usage.
        """
        check_interval = 10  # Check every 10 seconds

        while True:
            try:
                # Get per-CPU utilization
                per_cpu = psutil.cpu_percent(interval=1, percpu=True)
                avg_cpu = sum(per_cpu) / len(per_cpu)

                # Count how many CPUs are underutilized
                underutilized = sum(1 for cpu in per_cpu if cpu < 50)

                # Log current utilization
                logging.debug(f"CPU utilization: {avg_cpu:.1f}% (per-core: {per_cpu})")

                # If we have more than 2 underutilized CPUs and we're not at max workers
                if underutilized > 2 and self.n_workers < self.max_workers:
                    # Increase worker count more aggressively
                    old_workers = self.n_workers
                    increase = min(underutilized, self.max_workers - self.n_workers)
                    self.n_workers = min(self.n_workers + increase, self.max_workers)

                    if self.n_workers != old_workers:
                        logging.info(
                            f"Increasing workers from {old_workers} to {self.n_workers} due to {underutilized} underutilized CPUs"
                        )
                        self.worker_semaphore = asyncio.Semaphore(self.n_workers)

                # If average CPU is very high, check if we need to back off
                elif avg_cpu > 90 and self.n_workers > 1:
                    # Slightly reduce worker count to avoid system becoming unresponsive
                    old_workers = self.n_workers
                    self.n_workers = max(1, self.n_workers - 1)

                    if self.n_workers != old_workers:
                        logging.info(
                            f"Reducing workers from {old_workers} to {self.n_workers} due to high CPU usage ({avg_cpu:.1f}%)"
                        )
                        self.worker_semaphore = asyncio.Semaphore(self.n_workers)

            except Exception as e:
                logging.error(f"Error in CPU utilization monitor: {e}")

            await asyncio.sleep(check_interval)

    async def process_individuals_gpu(self, individuals, overrides_list, task_id):
        """Process individuals using GPU acceleration"""
        if not self.gpu_manager or not self.gpu_manager.available:
            logging.warning(
                "GPU processing requested but GPU is not available. Falling back to CPU."
            )
            return await self.process_individuals_cpu(
                individuals, overrides_list, task_id
            )

        results = []
        processed = 0
        valid_results = 0
        total = len(individuals)
        last_progress_update = time.time()

        try:
            # Determine GPU type and use appropriate implementation
            if self.gpu_manager.gpu_type == "nvidia":
                batch_results = await self.process_batch_nvidia_gpu(
                    individuals, overrides_list
                )
            elif self.gpu_manager.gpu_type == "amd":
                batch_results = await self.process_batch_amd_gpu(
                    individuals, overrides_list
                )
            else:
                logging.warning(
                    f"Unsupported GPU type: {self.gpu_manager.gpu_type}. Falling back to CPU."
                )
                return await self.process_individuals_cpu(
                    individuals, overrides_list, task_id
                )

            # Process results
            for result in batch_results:
                if result:
                    results.append(result)
                    valid_results += 1
                processed += 1

                # Send progress updates periodically
                current_time = time.time()
                if current_time - last_progress_update > 10:  # Every 10 seconds
                    await self.dealer_socket.send(
                        json.dumps(
                            {
                                "type": "progress",
                                "task_id": task_id,
                                "processed": processed,
                                "total": total,
                                "valid_results": valid_results,
                            }
                        ).encode()
                    )
                    last_progress_update = current_time

            return results

        except Exception as e:
            logging.error(f"Error in GPU processing: {e}")
            logging.info("Falling back to CPU processing")
            import traceback

            traceback.print_exc()

            # Fall back to CPU processing
            return await self.process_individuals_cpu(
                individuals, overrides_list, task_id
            )

    async def process_batch_nvidia_gpu(self, individuals, overrides_list):
        """Process a batch of individuals using NVIDIA GPU"""
        try:
            import cupy as cp
            from optimize import individual_to_config

            # Convert individuals to numpy array for batch processing
            individuals_array = np.array(individuals)

            # Transfer to GPU
            gpu_individuals = cp.array(individuals_array)

            # Create configs for each individual
            configs = []
            for individual in individuals:
                config = individual_to_config(
                    individual,
                    optimizer_overrides,
                    overrides_list,
                    template=self.config,
                )
                configs.append(config)

            # Process in batches if needed
            results = []
            batch_size = min(len(individuals), 10)  # Process 10 at a time to avoid OOM

            for i in range(0, len(individuals), batch_size):
                batch_individuals = individuals[i : i + batch_size]
                batch_configs = configs[i : i + batch_size]

                # Use thread pool for I/O-bound operations
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self.thread_pool,
                    self.evaluate_batch_gpu,
                    batch_individuals,
                    batch_configs,
                )

                results.extend(batch_results)

            return results

        except Exception as e:
            logging.error(f"Error in NVIDIA GPU batch processing: {e}")
            raise

    async def process_batch_amd_gpu(self, individuals, overrides_list):
        """Process a batch of individuals using AMD GPU"""
        try:
            import pyopencl as cl
            from optimize import individual_to_config

            # Create configs for each individual
            configs = []
            for individual in individuals:
                config = individual_to_config(
                    individual,
                    optimizer_overrides,
                    overrides_list,
                    template=self.config,
                )
                configs.append(config)

            # Process in batches
            results = []
            batch_size = min(len(individuals), 10)  # Process 10 at a time to avoid OOM

            for i in range(0, len(individuals), batch_size):
                batch_individuals = individuals[i : i + batch_size]
                batch_configs = configs[i : i + batch_size]

                # Use thread pool for I/O-bound operations
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self.thread_pool,
                    self.evaluate_batch_gpu_amd,
                    batch_individuals,
                    batch_configs,
                )

                results.extend(batch_results)

            return results

        except Exception as e:
            logging.error(f"Error in AMD GPU batch processing: {e}")
            raise

    def evaluate_batch_gpu(self, individuals, configs):
        """Evaluate a batch of individuals using NVIDIA GPU"""
        import cupy as cp
        from optimize import Evaluator
        import multiprocessing

        results = []

        for individual, config in zip(individuals, configs):
            # Create a queue for this process
            queue = multiprocessing.Queue()

            # Create evaluator
            evaluator = Evaluator(
                shared_memory_files=self.shared_memory_files,
                hlcvs_shapes=self.hlcvs_shapes,
                hlcvs_dtypes=self.hlcvs_dtypes,
                btc_usd_shared_memory_files=self.btc_usd_shared_memory_files,
                btc_usd_dtypes=self.btc_usd_dtypes,
                msss=self.msss,
                config=self.config,
                results_queue=queue,
                seen_hashes={},
                duplicate_counter={"count": 0},
                gpu_data=self.gpu_data if hasattr(self, "gpu_data") else None,
            )

            # Evaluate the individual
            evaluator.evaluate(individual, [])

            # Get result from queue
            result = queue.get()

            # Add individual to result
            result["individual"] = individual

            # Ensure config is present
            if "config" not in result:
                result["config"] = config

            results.append(result)

        return results

    def evaluate_batch_gpu_amd(self, individuals, configs):
        """Evaluate a batch of individuals using AMD GPU"""
        import pyopencl as cl
        from optimize import Evaluator
        import multiprocessing

        results = []

        for individual, config in zip(individuals, configs):
            # Create a queue for this process
            queue = multiprocessing.Queue()

            # Create evaluator
            evaluator = Evaluator(
                shared_memory_files=self.shared_memory_files,
                hlcvs_shapes=self.hlcvs_shapes,
                hlcvs_dtypes=self.hlcvs_dtypes,
                btc_usd_shared_memory_files=self.btc_usd_shared_memory_files,
                btc_usd_dtypes=self.btc_usd_dtypes,
                msss=self.msss,
                config=self.config,
                results_queue=queue,
                seen_hashes={},
                duplicate_counter={"count": 0},
                gpu_data=self.gpu_data if hasattr(self, "gpu_data") else None,
            )

            # Evaluate the individual
            evaluator.evaluate(individual, [])

            # Get result from queue
            result = queue.get()

            # Add individual to result
            result["individual"] = individual

            # Ensure config is present
            if "config" not in result:
                result["config"] = config

            results.append(result)

        return results

    # Add this function at module level (outside any class)
    def process_with_cached_evaluator(
        individual,
        overrides_list,
        config,
        shared_memory_files,
        hlcvs_shapes,
        hlcvs_dtypes,
        btc_usd_shared_memory_files,
        btc_usd_dtypes,
        msss,
        worker_id,
    ):
        """Process an individual with a cached evaluator (module-level function for pickling)"""
        # Try to optimize this process for CPU usage
        try:
            # Get current process
            process = psutil.Process()

            # Set CPU affinity if supported
            if hasattr(process, "cpu_affinity"):
                try:
                    # Calculate which core this worker should use
                    core_id = worker_id % psutil.cpu_count()
                    process.cpu_affinity([core_id])
                except Exception:
                    pass

            # Set process priority
            if sys.platform == "win32":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                # Unix-like systems
                process.nice(10)  # Lower priority
        except Exception:
            pass

        from optimize import Evaluator, individual_to_config, optimizer_overrides
        import multiprocessing

        # Create a queue for this process
        queue = multiprocessing.Queue()

        # Create evaluator
        evaluator = Evaluator(
            shared_memory_files=shared_memory_files,
            hlcvs_shapes=hlcvs_shapes,
            hlcvs_dtypes=hlcvs_dtypes,
            btc_usd_shared_memory_files=btc_usd_shared_memory_files,
            btc_usd_dtypes=btc_usd_dtypes,
            msss=msss,
            config=config,
            results_queue=queue,
            seen_hashes={},
            duplicate_counter={"count": 0},
        )

        # Evaluate the individual
        objectives = evaluator.evaluate(individual, overrides_list)

        # Get result from queue
        result = queue.get()

        # Add individual to result
        result["individual"] = individual

        # Ensure config is present
        if "config" not in result:
            result["config"] = individual_to_config(
                individual, optimizer_overrides, overrides_list, template=config
            )

        return result

    async def process_individual(
        self, individual, overrides_list, worker_semaphore, worker_id=0
    ):
        """Process a single individual with resource management"""
        try:
            # Check if we should pause due to high resource usage
            await self.resource_manager.wait_for_resources()

            # Acquire worker semaphore to limit concurrent evaluations
            async with worker_semaphore:
                # Log when we start processing this individual
                start_time = time.time()

                # Run the evaluation in a separate thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()

                # Run in process pool using the module-level function
                result = await loop.run_in_executor(
                    self.process_pool,
                    process_with_cached_evaluator,
                    individual,
                    overrides_list,
                    self.config,
                    self.shared_memory_files,
                    self.hlcvs_shapes,
                    self.hlcvs_dtypes,
                    self.btc_usd_shared_memory_files,
                    self.btc_usd_dtypes,
                    self.msss,
                    worker_id,
                )

                # Log completion time
                end_time = time.time()

                # Only log every 10th individual to reduce log spam
                if worker_id % 10 == 0:
                    logging.info(
                        f"Individual {worker_id} processed in {end_time - start_time:.2f} seconds."
                    )

                # Keep connection alive during long-running tasks
                if end_time - start_time > 60:  # If processing took more than a minute
                    try:
                        # Send a quick heartbeat to keep connection alive
                        await self.dealer_socket.send(
                            json.dumps({"type": "heartbeat"}).encode()
                        )
                    except Exception as e:
                        logging.debug(f"Error sending heartbeat during processing: {e}")

                return result

        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def heartbeat_loop(self):
        """Send periodic heartbeats to server with improved error handling"""
        heartbeat_interval = 10  # Send heartbeat every 10 seconds (more frequent)
        missed_heartbeats = 0

        while True:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory_percent = psutil.virtual_memory().percent

                # Get GPU info if available
                gpu_info = {}
                if self.use_gpu and self.gpu_manager and self.gpu_manager.available:
                    gpu_info = self.gpu_manager.get_gpu_info()

                resource_info = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "workers": self.n_workers,
                    "paused": self.resource_manager.paused,
                    "gpu": gpu_info,
                }

                # Send heartbeat
                await self.dealer_socket.send(
                    json.dumps(
                        {"type": "heartbeat", "resources": resource_info}
                    ).encode()
                )

                # Wait for acknowledgment with timeout
                try:
                    response = await asyncio.wait_for(
                        self.dealer_socket.recv(), timeout=5.0  # Shorter timeout
                    )

                    # Parse response
                    try:
                        response_json = response.decode()
                        response_data = json.loads(response_json)

                        # Check if response is compressed
                        if isinstance(response_data, dict) and response_data.get(
                            "compressed", False
                        ):
                            # Decompress the data
                            compressed_data = response_data.get("data")
                            if isinstance(compressed_data, str):
                                # Handle base64 encoded data if needed
                                import base64

                                compressed_data = base64.b64decode(compressed_data)

                            # Decompress
                            decompressed_data = zlib.decompress(compressed_data)
                            response_data = json.loads(decompressed_data.decode())
                    except Exception as e:
                        logging.error(f"Error parsing response: {e}")
                        continue

                    if response_data["type"] == "ack":
                        # Heartbeat acknowledged
                        missed_heartbeats = 0
                    elif response_data["type"] == "reregister":
                        # Server wants us to re-register
                        logging.info(
                            "Server requested re-registration during heartbeat"
                        )
                        await self.register_with_server()
                        missed_heartbeats = 0
                    elif response_data["type"] == "registered":
                        # This is a registration confirmation, not an expected heartbeat response
                        logging.info(
                            "Received registration confirmation during heartbeat"
                        )
                        self.config = response_data.get("config", self.config)
                        missed_heartbeats = 0
                        # Signal ready for tasks
                        await self.dealer_socket.send(
                            json.dumps({"type": "ready"}).encode()
                        )
                    elif response_data["type"] == "task":
                        # We got a task instead of an ack - handle it
                        logging.info("Received task during heartbeat")
                        self.current_task = response_data
                        self.is_processing_task = True
                        # Process the task in a separate task to not block heartbeats
                        asyncio.create_task(self.process_task(response_data))
                        missed_heartbeats = 0
                    else:
                        # Unexpected response
                        logging.warning(
                            f"Unexpected response to heartbeat: {response_data['type']}"
                        )
                except asyncio.TimeoutError:
                    # No acknowledgment received
                    missed_heartbeats += 1
                    logging.warning(
                        f"No heartbeat acknowledgment received (missed: {missed_heartbeats})"
                    )

                    if (
                        missed_heartbeats >= 2
                    ):  # Reduced from 3 to 2 for faster reconnection
                        # Try to reconnect after 2 consecutive missed heartbeats
                        logging.error(
                            "Connection appears to be lost, attempting to reconnect"
                        )
                        if await self.reconnect_to_server():
                            missed_heartbeats = 0
                        else:
                            # If reconnection fails, wait before trying again
                            await asyncio.sleep(5)  # Reduced wait time
                except asyncio.CancelledError:
                    # This is expected during shutdown
                    logging.info("Heartbeat loop cancelled")
                    raise
                except Exception as e:
                    logging.error(f"Error processing heartbeat response: {e}")
                    missed_heartbeats += 1
            except asyncio.CancelledError:
                # This is expected during shutdown
                logging.info("Heartbeat loop cancelled")
                raise
            except Exception as e:
                logging.error(f"Error in heartbeat loop: {e}")
                missed_heartbeats += 1

                if missed_heartbeats >= 2:  # Reduced from 3 to 2
                    # Try to reconnect after 2 consecutive errors
                    logging.error("Too many heartbeat errors, attempting to reconnect")
                    if await self.reconnect_to_server():
                        missed_heartbeats = 0
                    else:
                        # If reconnection fails, wait before trying again
                        await asyncio.sleep(5)  # Reduced wait time

            # Wait for next heartbeat
            await asyncio.sleep(heartbeat_interval)

    async def register_with_server(self):
        """Register with the server and initialize the evaluator"""
        try:
            # Get resource information
            resource_info = {
                "cpu_count": multiprocessing.cpu_count(),
                "workers": self.n_workers,
                "max_cpu_percent": self.max_cpu_percent,
                "max_memory_percent": self.max_memory_percent,
                "total_memory": psutil.virtual_memory().total,
            }

            # Add GPU information if available
            if self.use_gpu and self.gpu_manager:
                if self.gpu_manager.available:
                    gpu_info = self.gpu_manager.get_gpu_info()
                    resource_info["gpu"] = gpu_info
                    resource_info["gpu_available"] = True
                else:
                    resource_info["gpu_available"] = False

            # Send registration message
            await self.dealer_socket.send(
                json.dumps(
                    {
                        "type": "register",
                        "hostname": self.hostname,
                        "resources": resource_info,
                    }
                ).encode()
            )

            # Wait for registration confirmation with timeout
            try:
                message = await asyncio.wait_for(
                    self.dealer_socket.recv(), timeout=30.0
                )

                # Parse message
                try:
                    message_json = message.decode()
                    message_data = json.loads(message_json)

                    # Check if message is compressed
                    if isinstance(message_data, dict) and message_data.get(
                        "compressed", False
                    ):
                        # Decompress the data
                        compressed_data = message_data.get("data")
                        if isinstance(compressed_data, str):
                            # Handle base64 encoded data if needed
                            import base64

                            compressed_data = base64.b64decode(compressed_data)

                        # Decompress
                        decompressed_data = zlib.decompress(compressed_data)
                        message_data = json.loads(decompressed_data.decode())
                except Exception as e:
                    logging.error(f"Error parsing message: {e}")
                    return False

                if message_data["type"] == "registered":
                    logging.info(f"Successfully registered with server")
                    self.config = message_data["config"]

                    # Initialize evaluator if not already done
                    if not hasattr(self, "evaluator") or self.evaluator is None:
                        await self.initialize_evaluator()

                    # Signal ready for tasks
                    await self.dealer_socket.send(
                        json.dumps({"type": "ready"}).encode()
                    )
                    return True
                elif message_data["type"] == "task":
                    # Server sent a task immediately after registration
                    logging.info(f"Received task during registration")
                    self.config = message_data.get("config", self.config)

                    # Initialize evaluator if not already done
                    if not hasattr(self, "evaluator") or self.evaluator is None:
                        await self.initialize_evaluator()

                    # Process the task
                    self.current_task = message_data
                    self.is_processing_task = True
                    asyncio.create_task(self.process_task(message_data))
                    return True
                else:
                    logging.error(f"Failed to register with server: {message_data}")
                    return False
            except asyncio.TimeoutError:
                logging.error("Timeout waiting for registration confirmation")
                return False

        except Exception as e:
            logging.error(f"Error during registration: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Modify the status_listener method in the OptimizationClient class
    async def status_listener(self):
        """Listen for status broadcasts from server with reduced output frequency"""
        last_status_time = 0
        status_interval = 60  # Only show status every 60 seconds

        while True:
            try:
                message = await self.sub_socket.recv()

                # Parse message
                try:
                    message_json = message.decode()
                    status = json.loads(message_json)

                    # Check if message is compressed
                    if isinstance(status, dict) and status.get("compressed", False):
                        # Decompress the data
                        compressed_data = status.get("data")
                        if isinstance(compressed_data, str):
                            # Handle base64 encoded data if needed
                            import base64

                            compressed_data = base64.b64decode(compressed_data)

                        # Decompress
                        decompressed_data = zlib.decompress(compressed_data)
                        status = json.loads(decompressed_data.decode())
                except Exception as e:
                    logging.error(f"Error parsing status message: {e}")
                    continue

                # Only log status messages at a reasonable interval
                current_time = time.time()
                if (
                    status.get("type") == "status"
                    and current_time - last_status_time >= status_interval
                ):
                    last_status_time = current_time

                    # Format a more concise status message
                    clients = status.get("clients", 0)
                    pending = status.get("pending_tasks", 0)
                    completed = status.get("completed_tasks", 0)
                    pf_size = status.get("pareto_front_size", 0)
                    iteration = status.get("iteration", 0)

                    # More meaningful status message
                    logging.info(
                        f"Server Status | Clients: {clients} | Tasks: {pending} pending, {completed} done | "
                        f"Iterations: {iteration} | Pareto front: {pf_size} members"
                    )

                # Handle task progress updates
                elif status.get("type") == "task_progress":
                    task_id = status.get("task_id", "unknown")
                    processed = status.get("processed", 0)
                    total = status.get("total", 0)
                    valid = status.get("valid_results", 0)

                    # Only show progress if this is our task
                    if (
                        self.current_task
                        and self.current_task.get("task_id") == task_id
                    ):
                        # Clear the line and show progress
                        print(
                            f"\rTask progress: {processed}/{total} ({valid} valid) [{processed/total*100:.1f}%]",
                            end="",
                        )

            except Exception as e:
                logging.error(f"Error in status listener: {e}")

            await asyncio.sleep(1)  # Check frequently but don't busy-wait

    async def adaptive_worker_count(self):
        """Dynamically adjust worker count based on system load without recreating the process pool."""
        min_workers = 1
        max_workers = self.max_workers

        # On macOS, check if we're on battery power
        on_battery = False
        if sys.platform == "darwin":
            try:
                # Check if on battery on macOS
                power_info = os.popen("pmset -g batt").read()
                on_battery = "Battery Power" in power_info
                if on_battery:
                    logging.info(
                        "Running on battery power - will be more conservative with resources"
                    )
                    max_workers = max(
                        1, int(max_workers * 0.5)
                    )  # Reduce max workers on battery
            except:
                pass

        # On Windows, check if we're on battery power
        elif sys.platform == "win32":
            try:
                import ctypes

                status = ctypes.c_int()
                ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
                on_battery = not (status.value & 0x8)  # AC power flag
                if on_battery:
                    logging.info(
                        "Running on battery power - will be more conservative with resources"
                    )
                    max_workers = max(
                        1, int(max_workers * 0.5)
                    )  # Reduce max workers on battery
            except:
                pass

        # If using GPU, adjust worker count
        if self.use_gpu and self.gpu_manager and self.gpu_manager.available:
            # Reduce max workers to avoid CPU/GPU contention
            max_workers = max(1, int(max_workers * 0.8))
            logging.info(f"Adjusted max workers to {max_workers} due to GPU usage")

        while True:
            current_time = time.time()

            # Get current system metrics
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory().percent

            # Add to rolling averages
            self.last_cpu_readings.append(cpu)
            self.last_mem_readings.append(mem)

            # Keep only the most recent readings
            if len(self.last_cpu_readings) > self.max_readings:
                self.last_cpu_readings.pop(0)
            if len(self.last_mem_readings) > self.max_readings:
                self.last_mem_readings.pop(0)

            # Calculate smoothed values
            avg_cpu = sum(self.last_cpu_readings) / len(self.last_cpu_readings)
            avg_mem = sum(self.last_mem_readings) / len(self.last_mem_readings)

            # Check if we should adjust worker count (with cooldown)
            if current_time - self.last_worker_adjustment >= self.adjustment_cooldown:
                old_workers = self.n_workers

                # Detect if system is mostly idle
                system_idle = avg_cpu < 20 and not on_battery

                # Adjust worker count based on system load and aggressiveness
                if system_idle:
                    # System is idle, be more aggressive
                    target_workers = int(max_workers * 0.8 * self.aggressiveness)
                    self.n_workers = min(max(min_workers, target_workers), max_workers)
                elif (
                    avg_cpu < self.target_cpu_percent - 15
                    and avg_mem < self.target_memory_percent - 15
                ):
                    # System has plenty of headroom, increase workers
                    increase = max(1, int(self.n_workers * 0.2 * self.aggressiveness))
                    self.n_workers = min(self.n_workers + increase, max_workers)
                elif (
                    avg_cpu > self.max_cpu_percent - 5
                    or avg_mem > self.max_memory_percent - 5
                ):
                    # System is getting close to limits, decrease workers significantly
                    decrease = max(1, int(self.n_workers * 0.3))
                    self.n_workers = max(min_workers, self.n_workers - decrease)
                elif (
                    avg_cpu > self.target_cpu_percent + 5
                    or avg_mem > self.target_memory_percent + 5
                ):
                    # System is above target but below max, decrease workers slightly
                    decrease = max(1, int(self.n_workers * 0.1))
                    self.n_workers = max(min_workers, self.n_workers - decrease)

                # If worker count changed, update the semaphore
                if self.n_workers != old_workers:
                    logging.info(
                        f"Adjusting workers: {old_workers} → {self.n_workers} "
                        + f"(CPU: {avg_cpu:.1f}%, Mem: {avg_mem:.1f}%)"
                    )

                    # Create a new semaphore with the updated count
                    # This will affect new tasks but not interrupt current ones
                    self.worker_semaphore = asyncio.Semaphore(self.n_workers)

                    self.last_worker_adjustment = current_time

            # Check for foreground activity (different methods per OS)
            user_active = False

            if sys.platform == "darwin":
                try:
                    # Check if user is active on macOS by looking at window server CPU usage
                    for proc in psutil.process_iter(["name", "cpu_percent"]):
                        if (
                            proc.info["name"] == "WindowServer"
                            and proc.info["cpu_percent"] > 10
                        ):
                            user_active = True
                            break
                except:
                    pass
            elif sys.platform == "win32":
                try:
                    # Check if user is active on Windows
                    import ctypes

                    user_active = ctypes.windll.user32.GetForegroundWindow() != 0
                except:
                    pass
            elif sys.platform.startswith("linux"):
                try:
                    # Check if user is active on Linux by looking at X server CPU usage
                    for proc in psutil.process_iter(["name", "cpu_percent"]):
                        if (
                            proc.info["name"] in ["Xorg", "X", "wayland"]
                            and proc.info["cpu_percent"] > 5
                        ):
                            user_active = True
                            break
                except:
                    pass

            # If user is active, temporarily reduce worker count via semaphore
            if user_active and self.n_workers > min_workers:
                temp_workers = max(min_workers, int(self.n_workers * 0.5))
                if temp_workers != self.n_workers:
                    logging.info(
                        f"User activity detected - temporarily reducing workers to {temp_workers}"
                    )
                    # Just update the semaphore, not the actual worker count
                    self.worker_semaphore = asyncio.Semaphore(temp_workers)

            await asyncio.sleep(2)

    async def run(self):
        """Main client loop with robust error handling"""
        # Set up process pool
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        # Prewarm the process pool
        self.prewarm_process_pool()

        # Initialize worker semaphore
        self.worker_semaphore = asyncio.Semaphore(self.n_workers)

        # Connect to server
        server_host, server_port = self.server_address.split(":")
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())

        # Set socket options for more reliable connections
        self.dealer_socket.setsockopt(
            zmq.RECONNECT_IVL, 1000
        )  # Reconnect after 1 second
        self.dealer_socket.setsockopt(
            zmq.RECONNECT_IVL_MAX, 10000
        )  # Max 10 seconds between reconnects
        self.dealer_socket.setsockopt(
            zmq.LINGER, 0
        )  # Don't wait for unsent messages when closing

        self.dealer_socket.connect(f"tcp://{server_host}:{server_port}")
        self.sub_socket.connect(f"tcp://{server_host}:{int(server_port)+1}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Give ZMQ connections time to establish
        await asyncio.sleep(1)

        # Register with server
        registration_success = await self.register_with_server()
        if not registration_success:
            logging.error("Failed to register with server, exiting")
            self.cleanup()
            return

        # Start background tasks
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        status_task = asyncio.create_task(self.status_listener())
        adaptive_task = asyncio.create_task(self.adaptive_worker_count())
        cpu_monitor_task = asyncio.create_task(
            self.monitor_cpu_utilization()
        )  # Add this line

        # Track if we're currently processing a task
        self.is_processing_task = False

        try:
            # Main message handling loop
            connection_failures = 0
            while True:
                try:
                    # Receive message from server with timeout
                    message = await asyncio.wait_for(
                        self.dealer_socket.recv(), timeout=120  # 2 minute timeout
                    )

                    # Parse message
                    try:
                        message_json = message.decode()
                        message_data = json.loads(message_json)

                        # Check if message is compressed
                        if isinstance(message_data, dict) and message_data.get(
                            "compressed", False
                        ):
                            # Decompress the data
                            compressed_data = message_data.get("data")
                            if isinstance(compressed_data, str):
                                # Handle base64 encoded data if needed
                                import base64

                                compressed_data = base64.b64decode(compressed_data)

                            # Decompress
                            decompressed_data = zlib.decompress(compressed_data)
                            message_data = json.loads(decompressed_data.decode())
                    except Exception as e:
                        logging.error(f"Error parsing message: {e}")
                        continue

                    # Reset connection failure counter on successful message
                    connection_failures = 0

                    if message_data["type"] == "task":
                        # Process optimization task
                        self.current_task = message_data
                        self.is_processing_task = True
                        await self.process_task(message_data)
                        self.is_processing_task = False
                        self.current_task = None

                    elif message_data["type"] == "ack":
                        # Heartbeat acknowledgment, nothing to do
                        pass

                    elif message_data["type"] == "reregister":
                        # Server doesn't recognize us, re-register
                        logging.info("Server requested re-registration")
                        await self.register_with_server()

                    else:
                        logging.info(f"Received message: {message_data['type']}")

                except asyncio.TimeoutError:
                    # No message received within timeout period
                    connection_failures += 1
                    logging.warning(
                        f"No message received from server for 2 minutes (failures: {connection_failures})"
                    )

                    if connection_failures >= 3:
                        # Try to reconnect after 3 consecutive failures
                        logging.error(
                            "Connection appears to be lost, attempting to reconnect"
                        )
                        if await self.reconnect_to_server():
                            connection_failures = 0
                        else:
                            # If reconnection fails, wait before trying again
                            await asyncio.sleep(10)

                except asyncio.CancelledError:
                    logging.warning("Client operation was cancelled, cleaning up")
                    break

                except zmq.ZMQError as e:
                    # Handle ZMQ-specific errors
                    logging.error(f"ZMQ error: {e}")

                    if e.errno == zmq.EAGAIN:
                        # Resource temporarily unavailable (timeout)
                        connection_failures += 1
                        logging.warning(
                            f"ZMQ timeout (failures: {connection_failures})"
                        )

                        if connection_failures >= 3:
                            # Try to reconnect after 3 consecutive failures
                            logging.error(
                                "Connection appears to be lost, attempting to reconnect"
                            )
                            if await self.reconnect_to_server():
                                connection_failures = 0
                            else:
                                # If reconnection fails, wait before trying again
                                await asyncio.sleep(10)
                    else:
                        # Other ZMQ errors - try to reconnect
                        logging.error("Attempting to reconnect due to ZMQ error")
                        if await self.reconnect_to_server():
                            connection_failures = 0
                        else:
                            # If reconnection fails, wait before trying again
                            await asyncio.sleep(10)

                except Exception as e:
                    logging.error(f"Error in client loop: {e}")
                    import traceback

                    traceback.print_exc()

                    # If we were processing a task, report the error
                    if self.current_task and self.is_processing_task:
                        try:
                            await self.dealer_socket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "task_id": self.current_task["task_id"],
                                        "error": str(e),
                                    }
                                ).encode()
                            )

                            # Signal ready for more tasks
                            await self.dealer_socket.send(
                                json.dumps({"type": "ready"}).encode()
                            )

                            self.current_task = None
                            self.is_processing_task = False
                        except:
                            pass

                    # Wait a bit before continuing
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logging.warning("Client operation was cancelled, cleaning up")

        except Exception as e:
            logging.error(f"Unhandled exception in client run loop: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Cancel background tasks
            for task in [heartbeat_task, status_task, adaptive_task, cpu_monitor_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Clean up resources
            self.cleanup()

    def cleanup(self):
        """Clean up resources before exit"""
        logging.info("Cleaning up resources...")

        # Shutdown process pool
        if hasattr(self, "process_pool"):
            logging.info("Shutting down process pool...")
            self.process_pool.shutdown(wait=False)

        # Shutdown thread pool
        if hasattr(self, "thread_pool"):
            logging.info("Shutting down thread pool...")
            self.thread_pool.shutdown(wait=False)

        # Clean up GPU resources
        if hasattr(self, "gpu_manager") and self.gpu_manager:
            logging.info("Cleaning up GPU resources...")
            self.gpu_manager.cleanup()

        # Remove shared memory files
        for shared_memory_file in self.shared_memory_files.values():
            if shared_memory_file and os.path.exists(shared_memory_file):
                logging.info(f"Removing shared memory file: {shared_memory_file}")
                try:
                    os.unlink(shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing shared memory file: {e}")

        # Remove BTC USD shared memory files
        for shared_memory_file in self.btc_usd_shared_memory_files.values():
            if shared_memory_file and os.path.exists(shared_memory_file):
                logging.info(
                    f"Removing BTC USD shared memory file: {shared_memory_file}"
                )
                try:
                    os.unlink(shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing BTC USD shared memory file: {e}")

        logging.info("Cleanup complete")


class GPUManager:
    """Manages GPU resources and operations"""

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.available = False
        self.gpu_type = None
        self.context = None
        self.queue = None
        self.device = None
        self.memory_total = 0
        self.memory_free = 0
        self.name = "Unknown GPU"

        # Try to initialize GPU
        self.initialize()

    def initialize(self):
        """Initialize GPU resources"""
        # First try NVIDIA GPU with CuPy
        if self.try_initialize_nvidia():
            self.gpu_type = "nvidia"
            self.available = True
            logging.info(
                f"Successfully initialized NVIDIA GPU {self.gpu_id}: {self.name}"
            )
            return True

        # Then try AMD GPU with PyOpenCL
        if self.try_initialize_amd():
            self.gpu_type = "amd"
            self.available = True
            logging.info(f"Successfully initialized AMD GPU {self.gpu_id}: {self.name}")
            return True

        logging.warning("No compatible GPU found or GPU libraries not installed")
        return False

    def try_initialize_nvidia(self):
        """Try to initialize NVIDIA GPU using CuPy"""
        try:
            import cupy as cp

            # Check if CuPy can see any GPUs
            if cp.cuda.runtime.getDeviceCount() == 0:
                logging.warning("CuPy found no CUDA devices")
                return False

            # Set device
            cp.cuda.Device(self.gpu_id).use()

            # Get device properties
            self.name = cp.cuda.runtime.getDeviceProperties(self.gpu_id)[
                "name"
            ].decode()
            free, total = cp.cuda.runtime.memGetInfo()
            self.memory_free = free
            self.memory_total = total

            logging.info(
                f"NVIDIA GPU {self.gpu_id}: {self.name}, {total / (1024**3):.2f} GB total, {free / (1024**3):.2f} GB free"
            )

            return True
        except ImportError:
            logging.info("CuPy not installed, cannot use NVIDIA GPU")
            return False
        except Exception as e:
            logging.warning(f"Error initializing NVIDIA GPU: {e}")
            return False

    def try_initialize_amd(self):
        """Try to initialize AMD GPU using PyOpenCL"""
        try:
            import pyopencl as cl

            # Get platforms
            platforms = cl.get_platforms()
            if not platforms:
                logging.warning("PyOpenCL found no platforms")
                return False

            # Find AMD platform
            amd_platform = None
            for platform in platforms:
                if "AMD" in platform.name:
                    amd_platform = platform
                    break

            if not amd_platform:
                # Try any platform with GPU devices
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        amd_platform = platform
                        break

            if not amd_platform:
                logging.warning("No platform with GPU devices found")
                return False

            # Get GPU devices
            devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                logging.warning(f"No GPU devices found on platform {amd_platform.name}")
                return False

            # Select device
            if self.gpu_id >= len(devices):
                logging.warning(f"GPU ID {self.gpu_id} out of range, using device 0")
                self.gpu_id = 0

            self.device = devices[self.gpu_id]
            self.name = self.device.name
            self.memory_total = self.device.global_mem_size

            # Create context and queue
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)

            logging.info(
                f"AMD GPU {self.gpu_id}: {self.name}, {self.memory_total / (1024**3):.2f} GB total"
            )

            return True
        except ImportError:
            logging.info("PyOpenCL not installed, cannot use AMD GPU")
            return False
        except Exception as e:
            logging.warning(f"Error initializing AMD GPU: {e}")
            return False

    def get_gpu_info(self):
        """Get current GPU information"""
        info = {
            "available": self.available,
            "type": self.gpu_type,
            "name": self.name,
            "memory_total": self.memory_total,
        }

        # Get current memory usage
        if self.available:
            if self.gpu_type == "nvidia":
                try:
                    import cupy as cp

                    free, total = cp.cuda.runtime.memGetInfo()
                    info["memory_free"] = free
                    info["memory_used"] = total - free
                    info["memory_percent"] = ((total - free) / total) * 100
                except Exception as e:
                    logging.error(f"Error getting NVIDIA GPU memory info: {e}")
            elif self.gpu_type == "amd":
                # AMD doesn't provide easy access to current memory usage
                # We'll just use the total memory
                info["memory_free"] = "unknown"
                info["memory_used"] = "unknown"
                info["memory_percent"] = "unknown"

        return info

    def cleanup(self):
        """Clean up GPU resources"""
        if not self.available:
            return

        try:
            if self.gpu_type == "nvidia":
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceReset()
                logging.info("NVIDIA GPU resources cleaned up")
            elif self.gpu_type == "amd":
                # Release OpenCL resources
                if self.queue:
                    self.queue.finish()
                self.queue = None
                self.context = None
                logging.info("AMD GPU resources cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up GPU resources: {e}")


async def main():
    """Main entry point with improved error handling"""
    # Ensure Rust code is compiled
    manage_rust_compilation()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Distributed optimization for Passivbot with GPU acceleration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["server", "client"],
        help="Run in server or client mode",
    )

    # Server-specific arguments
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (server mode only)"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port to listen on (server mode only)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of individuals to send in each task (server mode only)",
    )

    # Client-specific arguments
    parser.add_argument(
        "--server",
        type=str,
        help="Server address in format host:port (client mode only)",
    )
    parser.add_argument(
        "--max-cpu",
        type=int,
        default=70,
        help="Maximum CPU usage percentage (client mode only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimization from checkpoint if available (server mode only)",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=80,
        help="Maximum memory usage percentage (client mode only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto, client mode only)",
    )
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=1.0,
        help="Aggressiveness factor (0.1-1.0, default 1.0, lower = more gentle on system)",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="low",
        choices=["low", "normal", "high"],
        help="Process priority (client mode only)",
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Enable aggressive memory optimization (client mode only)",
    )

    # GPU-specific arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available (client mode only)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (client mode only)",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="auto",
        choices=["auto", "nvidia", "amd", "none"],
        help="GPU type to use (auto, nvidia, amd, none) (client mode only)",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=85,
        help="Maximum GPU memory usage percentage (client mode only)",
    )
    parser.add_argument(
        "--hybrid-mode",
        action="store_true",
        help="Use both GPU and CPU for processing (client mode only)",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=10,
        help="Batch size for GPU processing in hybrid mode (client mode only)",
    )
    parser.add_argument(
        "--cpu-batch-size",
        type=int,
        default=10,
        help="Batch size for CPU processing in hybrid mode (client mode only)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=300,  # 5 minutes in seconds
        help="Interval in seconds between saving checkpoints (server mode only)",
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Enable compression for large messages (client mode only)",
    )

    args = parser.parse_args()

    # Create and run the appropriate component
    client = None
    try:
        if args.mode == "server":
            if not args.config:
                parser.error("--config is required in server mode")

            server = OptimizationServer(args)
            await server.run()
        else:  # client mode
            if not args.server:
                parser.error("--server is required in client mode")

            client = OptimizationClient(args)
            try:
                await client.run()
            except asyncio.CancelledError:
                logging.info("Client run was cancelled, performing cleanup")
                # This is expected during shutdown, not an error
                pass
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        if args.mode == "client" and client is not None:
            try:
                client.cleanup()
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")

        logging.info("Exiting...")


if __name__ == "__main__":
    # Import zlib for compression
    import zlib

    # Initialize thread pool for I/O-bound operations
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    asyncio.run(main())
