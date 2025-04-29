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
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import cpu_count

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

# Check for GPU support
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.signal as cusignal
    from numba import cuda

    if cuda.is_available():
        GPU_AVAILABLE = True
        logging.info("CUDA GPU support detected and enabled")
    else:
        logging.warning("CUDA libraries found but no GPU detected")
except ImportError:
    logging.warning("GPU acceleration libraries not found, running in CPU-only mode")


class ResourceManager:
    """Manages CPU and memory resources for optimization"""

    def __init__(self, max_cpu_percent=70, max_memory_percent=80, check_interval=5):
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

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)

        # Check memory usage
        memory_percent = psutil.virtual_memory().percent

        # Determine if we should pause
        if (
            cpu_percent > self.max_cpu_percent
            or memory_percent > self.max_memory_percent
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


class GPUResourceManager(ResourceManager):
    """Manages GPU resources for optimization"""

    def __init__(
        self,
        max_cpu_percent=70,
        max_memory_percent=80,
        max_gpu_percent=80,
        check_interval=5,
    ):
        super().__init__(max_cpu_percent, max_memory_percent, check_interval)
        self.max_gpu_percent = max_gpu_percent

    def should_pause(self):
        """Check if processing should be paused due to high resource usage"""
        # First check CPU and memory
        cpu_memory_pause = super().should_pause()
        if cpu_memory_pause:
            return True

        # Then check GPU if available
        if GPU_AVAILABLE:
            try:
                # Get GPU memory usage
                gpu_memory_used = cp.cuda.runtime.memGetInfo()[0]
                gpu_memory_total = cp.cuda.runtime.memGetInfo()[1]
                gpu_percent = (gpu_memory_used / gpu_memory_total) * 100

                if gpu_percent > self.max_gpu_percent:
                    if not self.paused:
                        logging.info(
                            f"Pausing due to high GPU usage: {gpu_percent:.1f}%"
                        )
                        self.paused = True
                    return True
            except Exception as e:
                logging.warning(f"Error checking GPU usage: {e}")

        return self.paused


class GPUBacktester:
    """GPU-accelerated backtesting implementation"""

    def __init__(self, config, hlcvs_dict, btc_usd_data_dict):
        self.config = config
        self.hlcvs_dict = hlcvs_dict
        self.btc_usd_data_dict = btc_usd_data_dict
        self.device_id = 0
        self.batch_size = 32  # Number of individuals to evaluate in parallel

        # Transfer data to GPU
        self.setup_gpu()

    def setup_gpu(self):
        """Transfer market data to GPU memory"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but not available")

        # Set device
        cp.cuda.Device(self.device_id).use()

        # Transfer HLCV data to GPU
        self.gpu_hlcvs = {}
        self.gpu_btc_usd = {}

        for exchange, hlcvs in self.hlcvs_dict.items():
            logging.info(f"Transferring {exchange} data to GPU memory...")
            self.gpu_hlcvs[exchange] = cp.asarray(hlcvs)
            self.gpu_btc_usd[exchange] = cp.asarray(self.btc_usd_data_dict[exchange])

        logging.info("Data transfer to GPU complete")

    def evaluate_batch(self, individuals, overrides_list):
        """Evaluate a batch of individuals in parallel on GPU"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but not available")

        # Prepare configs for all individuals
        configs = []
        for individual in individuals:
            config = individual_to_config(
                individual, optimizer_overrides, overrides_list, template=self.config
            )
            configs.append(config)

        # Run GPU-accelerated backtest
        results = self._run_gpu_backtest(configs)

        return results

    def _run_gpu_backtest(self, configs):
        """
        Run backtest for multiple configs in parallel on GPU
        This is a simplified implementation - a full implementation would
        require porting the Rust backtest logic to CUDA
        """
        # This is where we would implement the GPU-accelerated backtest
        # For now, we'll use a placeholder implementation that returns dummy results

        results = []
        for config in configs:
            # In a real implementation, this would be replaced with actual GPU computation
            result = {
                "config": config,
                "analyses_combined": {
                    "adg_mean": np.random.random() * 0.01,
                    "drawdown_worst_mean": np.random.random() * 0.2,
                    "w_0": np.random.random()
                    * -0.01,  # Placeholder for actual objectives
                    "w_1": np.random.random() * 0.2,
                },
            }
            results.append(result)

        return results

    def cleanup(self):
        """Free GPU memory"""
        if GPU_AVAILABLE:
            for exchange in self.gpu_hlcvs:
                del self.gpu_hlcvs[exchange]
                del self.gpu_btc_usd[exchange]

            # Force memory release
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


@cuda.jit
def calculate_emas_kernel(prices, ema_spans, output):
    """CUDA kernel to calculate EMAs for multiple spans in parallel"""
    # This is a simplified example of a CUDA kernel for EMA calculation
    # In a real implementation, this would be more complex

    idx = cuda.grid(1)
    if idx < prices.shape[0]:
        for i in range(ema_spans.shape[0]):
            span = ema_spans[i]
            alpha = 2.0 / (span + 1.0)

            if idx == 0:
                output[idx, i] = prices[idx]
            else:
                output[idx, i] = (
                    alpha * prices[idx] + (1.0 - alpha) * output[idx - 1, i]
                )


class DistributedOptimizer:
    """Base class for both server and client implementations"""

    def __init__(self, args):
        self.args = args
        self.node_id = str(uuid.uuid4())[:8]
        self.context = zmq.asyncio.Context()
        self.use_gpu = args.use_gpu and GPU_AVAILABLE

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

        # ZMQ sockets
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.pub_socket = self.context.socket(zmq.PUB)

        # Task generation parameters
        self.batch_size = args.batch_size
        self.population_size = 100  # Will be updated from config

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

    async def handle_client_message(self, client_id, message):
        """Process messages from clients"""
        # Decode client_id if it's bytes
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8")

        msg_type = message.get("type")

        if msg_type == "register":
            # New client registration
            hostname = message.get("hostname", "unknown")
            has_gpu = message.get("has_gpu", False)
            resource_info = message.get("resources", {})

            self.clients[client_id] = {
                "hostname": hostname,
                "has_gpu": has_gpu,
                "resources": resource_info,
                "last_seen": time.time(),
                "status": "idle",
                "tasks_completed": 0,
            }
            logging.info(
                f"New client registered: {hostname} ({client_id}) - GPU: {has_gpu}"
            )
            await self.router_socket.send_multipart(
                [
                    client_id.encode() if isinstance(client_id, str) else client_id,
                    json.dumps({"type": "registered", "config": self.config}).encode(),
                ]
            )

        elif msg_type == "heartbeat":
            # Update client's last seen timestamp
            if client_id in self.clients:
                self.clients[client_id]["last_seen"] = time.time()
                # Update resource info if provided
                if "resources" in message:
                    self.clients[client_id]["resources"] = message["resources"]
                await self.router_socket.send_multipart(
                    [
                        client_id.encode() if isinstance(client_id, str) else client_id,
                        json.dumps({"type": "ack"}).encode(),
                    ]
                )

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

    async def send_tasks_to_client(self, client_id):
        """Send optimization tasks to a client"""
        if not self.population:
            # No more tasks to send
            return

        # Ensure client_id is properly decoded
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8")

        # Get batch of individuals to evaluate
        batch_size = min(self.batch_size, len(self.population))
        individuals = self.population[:batch_size]
        self.population = self.population[batch_size:]

        # Create task
        task_id = str(uuid.uuid4())
        self.pending_tasks[task_id] = individuals

        # Send task to client
        task_message = {
            "type": "task",
            "task_id": task_id,
            "individuals": individuals,
            "param_bounds": self.param_bounds,
        }

        await self.router_socket.send_multipart(
            [
                client_id.encode() if isinstance(client_id, str) else client_id,
                json.dumps(task_message).encode(),
            ]
        )

        # Update client status
        self.clients[client_id]["status"] = "busy"
        logging.info(
            f"Sent task {task_id} with {len(individuals)} individuals to client {client_id}"
        )

    async def process_results(self, results):
        """Process optimization results from clients"""
        for result in results:
            individual = result["individual"]
            config = result["config"]
            analyses_combined = result["analyses_combined"]

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

        for client_id, client_info in self.clients.items():
            # Check if client hasn't sent a heartbeat in 60 seconds
            if current_time - client_info["last_seen"] > 60:
                logging.warning(
                    f"Client {client_id} ({client_info['hostname']}) appears to be disconnected"
                )
                disconnected_clients.append(client_id)

                # If client had pending tasks, return them to the queue
                for task_id, individuals in list(self.pending_tasks.items()):
                    if task_id.startswith(client_id):
                        logging.info(f"Returning task {task_id} to queue")
                        self.population.extend(individuals)
                        self.pending_tasks.pop(task_id)

        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.clients.pop(client_id)

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

        # Main message handling loop
        while True:
            try:
                # Receive message from clients
                client_id, message = await self.router_socket.recv_multipart()
                message = json.loads(message.decode())

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

        # Resource management
        self.max_cpu_percent = args.max_cpu
        self.max_memory_percent = args.max_memory
        self.max_gpu_percent = args.max_gpu if self.use_gpu else 0
        self.n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

        # Initialize resource manager
        if self.use_gpu:
            self.resource_manager = GPUResourceManager(
                max_cpu_percent=self.max_cpu_percent,
                max_memory_percent=self.max_memory_percent,
                max_gpu_percent=self.max_gpu_percent,
            )
        else:
            self.resource_manager = ResourceManager(
                max_cpu_percent=self.max_cpu_percent,
                max_memory_percent=self.max_memory_percent,
            )

        # ZMQ sockets
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.sub_socket = self.context.socket(zmq.SUB)

        # Task processing
        self.current_task = None
        self.config = None
        self.evaluator = None
        self.gpu_backtester = None
        self.shared_memory_files = {}
        self.hlcvs_dict = {}
        self.btc_usd_data_dict = {}

    async def setup(self):
        """Connect to server and register"""
        # Set client ID in socket
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())

        # Connect to server
        server_host, server_port = self.server_address.split(":")
        self.dealer_socket.connect(f"tcp://{server_host}:{server_port}")
        self.sub_socket.connect(f"tcp://{server_host}:{int(server_port)+1}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Prepare resource info
        resource_info = {
            "cpu_count": cpu_count(),
            "workers": self.n_workers,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
            "total_memory": psutil.virtual_memory().total,
        }

        if self.use_gpu:
            try:
                gpu_info = {
                    "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    "gpu_memory": cp.cuda.runtime.memGetInfo()[1],
                    "max_gpu_percent": self.max_gpu_percent,
                }
                resource_info.update(gpu_info)
            except Exception as e:
                logging.warning(f"Could not get GPU info: {e}")

        # Register with server
        await self.dealer_socket.send(
            json.dumps(
                {
                    "type": "register",
                    "hostname": self.hostname,
                    "has_gpu": self.use_gpu,
                    "resources": resource_info,
                }
            ).encode()
        )

        # Wait for registration confirmation
        message = await self.dealer_socket.recv()
        message = json.loads(message.decode())

        if message["type"] == "registered":
            logging.info(f"Successfully registered with server")
            self.config = message["config"]

            # Initialize evaluator
            await self.initialize_evaluator()

            # Signal ready for tasks
            await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())
        else:
            logging.error(f"Failed to register with server: {message}")
            sys.exit(1)

    async def initialize_evaluator(self):
        """Initialize the optimization evaluator"""
        logging.info("Initializing evaluator...")

        # Prepare data for each exchange
        self.hlcvs_dict = {}
        self.shared_memory_files = {}
        hlcvs_shapes = {}
        hlcvs_dtypes = {}
        msss = {}

        # Store per-exchange BTC arrays
        self.btc_usd_data_dict = {}
        btc_usd_shared_memory_files = {}
        btc_usd_dtypes = {}

        self.config["backtest"]["coins"] = {}

        if self.config["backtest"]["combine_ohlcvs"]:
            exchange = "combined"
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = (
                await prepare_hlcvs_mss(self.config, exchange)
            )
            self.config["backtest"]["coins"][exchange] = coins
            self.hlcvs_dict[exchange] = hlcvs
            hlcvs_shapes[exchange] = hlcvs.shape
            hlcvs_dtypes[exchange] = hlcvs.dtype
            msss[exchange] = mss

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
            btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                self.btc_usd_data_dict[exchange]
            )
            btc_usd_dtypes[exchange] = self.btc_usd_data_dict[exchange].dtype

        else:
            tasks = {}
            for exchange in self.config["backtest"]["exchanges"]:
                tasks[exchange] = asyncio.create_task(
                    prepare_hlcvs_mss(self.config, exchange)
                )

            for exchange in self.config["backtest"]["exchanges"]:
                coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices = (
                    await tasks[exchange]
                )
                self.config["backtest"]["coins"][exchange] = coins
                self.hlcvs_dict[exchange] = hlcvs
                hlcvs_shapes[exchange] = hlcvs.shape
                hlcvs_dtypes[exchange] = hlcvs.dtype
                msss[exchange] = mss

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
                btc_usd_shared_memory_files[exchange] = create_shared_memory_file(
                    self.btc_usd_data_dict[exchange]
                )
                btc_usd_dtypes[exchange] = self.btc_usd_data_dict[exchange].dtype

        # Create results queue for evaluator
        self.results_queue = asyncio.Queue()

        # Import Evaluator class
        from optimize import Evaluator

        # Initialize evaluator with shared memory files
        self.evaluator = Evaluator(
            shared_memory_files=self.shared_memory_files,
            hlcvs_shapes=hlcvs_shapes,
            hlcvs_dtypes=hlcvs_dtypes,
            btc_usd_shared_memory_files=btc_usd_shared_memory_files,
            btc_usd_dtypes=btc_usd_dtypes,
            msss=msss,
            config=self.config,
            results_queue=self.results_queue,
            seen_hashes={},
            duplicate_counter={"count": 0},
        )

        # Initialize GPU backtester if GPU is available and requested
        if self.use_gpu:
            try:
                self.gpu_backtester = GPUBacktester(
                    self.config, self.hlcvs_dict, self.btc_usd_data_dict
                )
                logging.info("GPU backtester initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize GPU backtester: {e}")
                self.use_gpu = False

        logging.info("Evaluator initialization complete")

    async def process_task(self, task):
        """Process an optimization task"""
        task_id = task["task_id"]
        individuals = task["individuals"]
        overrides_list = self.config.get("optimize", {}).get("enable_overrides", [])

        logging.info(f"Processing task {task_id} with {len(individuals)} individuals")

        results = []

        # Create a worker pool based on configured number of workers
        worker_semaphore = asyncio.Semaphore(self.n_workers)

        # Use GPU backtester if available
        if self.use_gpu and self.gpu_backtester:
            try:
                # Check resource usage before GPU processing
                await self.resource_manager.wait_for_resources()

                # Process batch on GPU
                batch_results = self.gpu_backtester.evaluate_batch(
                    individuals, overrides_list
                )

                # Add individual to each result for tracking
                for i, result in enumerate(batch_results):
                    result["individual"] = individuals[i]
                    results.append(result)

                logging.info(f"Processed {len(batch_results)} individuals on GPU")

            except Exception as e:
                logging.error(f"Error in GPU evaluation, falling back to CPU: {e}")
                import traceback

                traceback.print_exc()

                # Fall back to CPU evaluation
                tasks = []
                for individual in individuals:
                    tasks.append(
                        self.process_individual(
                            individual, overrides_list, worker_semaphore
                        )
                    )

                # Wait for all individuals to be processed
                for result in await asyncio.gather(*tasks):
                    if result:
                        results.append(result)
        else:
            # CPU-only evaluation with resource management
            tasks = []
            for individual in individuals:
                tasks.append(
                    self.process_individual(
                        individual, overrides_list, worker_semaphore
                    )
                )

            # Wait for all individuals to be processed
            for result in await asyncio.gather(*tasks):
                if result:
                    results.append(result)

        # Send results back to server
        await self.dealer_socket.send(
            json.dumps(
                {"type": "result", "task_id": task_id, "results": results}
            ).encode()
        )

        # Signal ready for more tasks
        await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())

        logging.info(f"Completed task {task_id}")

    async def process_individual(self, individual, overrides_list, worker_semaphore):
        """Process a single individual with resource management"""
        try:
            # Check if we should pause due to high resource usage
            await self.resource_manager.wait_for_resources()

            # Acquire worker semaphore to limit concurrent evaluations
            async with worker_semaphore:
                # Evaluate individual
                objectives = self.evaluator.evaluate(individual, overrides_list)

                # Get result from queue
                result = await self.results_queue.get()

                # Add individual to result for tracking
                result["individual"] = individual

                return result

        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def heartbeat_loop(self):
        """Send periodic heartbeats to server"""
        while True:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory_percent = psutil.virtual_memory().percent

                resource_info = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "workers": self.n_workers,
                    "paused": self.resource_manager.paused,
                }

                # Add GPU info if available
                if self.use_gpu:
                    try:
                        gpu_memory_used = cp.cuda.runtime.memGetInfo()[0]
                        gpu_memory_total = cp.cuda.runtime.memGetInfo()[1]
                        gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                        resource_info["gpu_percent"] = gpu_percent
                    except Exception as e:
                        logging.warning(f"Error getting GPU info: {e}")

                await self.dealer_socket.send(
                    json.dumps(
                        {"type": "heartbeat", "resources": resource_info}
                    ).encode()
                )
            except Exception as e:
                logging.error(f"Error sending heartbeat: {e}")

            await asyncio.sleep(30)  # Send heartbeat every 30 seconds

    async def status_listener(self):
        """Listen for status broadcasts from server"""
        while True:
            try:
                message = await self.sub_socket.recv()
                status = json.loads(message.decode())

                if status.get("type") == "status":
                    logging.info(
                        f"Server status: {status['clients']} clients, "
                        f"{status['pending_tasks']} pending tasks, "
                        f"{status['completed_tasks']} completed tasks, "
                        f"PF size: {status['pareto_front_size']}"
                    )
            except Exception as e:
                logging.error(f"Error in status listener: {e}")

            await asyncio.sleep(1)  # Check frequently but don't busy-wait

    async def adaptive_worker_count(self):
        min_workers = 1
        max_workers = cpu_count()
        target_cpu = self.max_cpu_percent
        target_gpu = self.max_gpu_percent if self.use_gpu else 100
        while True:
            cpu = psutil.cpu_percent(interval=1)
            gpu_percent = 0
            if self.use_gpu:
                try:
                    gpu_mem_used, gpu_mem_total = cp.cuda.runtime.memGetInfo()
                    gpu_percent = 100 * (1 - gpu_mem_used / gpu_mem_total)
                except Exception:
                    gpu_percent = 0
            # Adjust based on both CPU and GPU
            if cpu < target_cpu - 10 and gpu_percent < target_gpu - 10:
                self.n_workers = min(self.n_workers + 1, max_workers)
            elif cpu > target_cpu + 5 or gpu_percent > target_gpu + 5:
                self.n_workers = max(self.n_workers - 1, min_workers)
            await asyncio.sleep(2)

    async def run(self):
        """Main client loop"""
        await self.setup()

        # Start background tasks
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        status_task = asyncio.create_task(self.status_listener())
        adaptive_task = asyncio.create_task(self.adaptive_worker_count())

        # Main message handling loop
        while True:
            try:
                # Receive message from server
                message = await self.dealer_socket.recv()
                message = json.loads(message.decode())

                if message["type"] == "task":
                    # Process optimization task
                    self.current_task = message
                    await self.process_task(message)
                elif message["type"] == "ack":
                    # Heartbeat acknowledgment, nothing to do
                    pass
                else:
                    logging.info(f"Received message: {message['type']}")

            except Exception as e:
                logging.error(f"Error in client loop: {e}")
                import traceback

                traceback.print_exc()

                # If we were processing a task, report the error
                if self.current_task:
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
                    except:
                        pass

    def cleanup(self):
        """Clean up resources before exit"""
        # Clean up GPU resources if used
        if self.use_gpu and self.gpu_backtester:
            try:
                self.gpu_backtester.cleanup()
            except Exception as e:
                logging.error(f"Error cleaning up GPU resources: {e}")

        # Remove shared memory files
        for shared_memory_file in self.shared_memory_files.values():
            if shared_memory_file and os.path.exists(shared_memory_file):
                logging.info(f"Removing shared memory file: {shared_memory_file}")
                try:
                    os.unlink(shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing shared memory file: {e}")


class GPUGeneticOperators:
    """GPU-accelerated genetic algorithm operators"""

    def __init__(self, param_bounds, sig_digits=6):
        self.param_bounds = param_bounds
        self.sig_digits = sig_digits
        self.bounds = [(low, high) for param, (low, high) in param_bounds.items()]

        # Transfer bounds to GPU
        if GPU_AVAILABLE:
            self.gpu_bounds = cp.array(self.bounds)

    def crossover_gpu(self, population, crossover_rate=0.7):
        """Perform crossover operations on GPU"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but not available")

        # Convert population to GPU array
        gpu_pop = cp.array(population)
        pop_size, n_params = gpu_pop.shape

        # Generate random pairs for crossover
        pairs = cp.random.permutation(pop_size).reshape(-1, 2)

        # Generate crossover points
        crossover_points = cp.random.randint(1, n_params, size=pairs.shape[0])

        # Generate crossover mask
        should_crossover = cp.random.random(pairs.shape[0]) < crossover_rate

        # Perform crossover on GPU
        new_population = gpu_pop.copy()

        # This would be implemented as a CUDA kernel in a real application
        # For now, we'll use a simple loop
        for i, (a, b) in enumerate(pairs):
            if should_crossover[i]:
                point = crossover_points[i]
                temp = new_population[a, point:].copy()
                new_population[a, point:] = new_population[b, point:]
                new_population[b, point:] = temp

        # Round values and ensure bounds
        for i in range(n_params):
            low, high = self.bounds[i]
            if low != high:
                new_population[:, i] = cp.clip(new_population[:, i], low, high)

        # Round to significant digits
        if self.sig_digits > 0:
            # This would be a custom CUDA kernel in a real implementation
            new_population_cpu = new_population.get()
            for i in range(pop_size):
                for j in range(n_params):
                    if self.bounds[j][0] != self.bounds[j][1]:
                        new_population_cpu[i, j] = round_floats(
                            new_population_cpu[i, j], self.sig_digits
                        )
            new_population = cp.array(new_population_cpu)

        return new_population.get()

    def mutate_gpu(self, population, mutation_rate=0.1, mutation_strength=0.1):
        """Perform mutation operations on GPU"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but not available")

        # Convert population to GPU array
        gpu_pop = cp.array(population)
        pop_size, n_params = gpu_pop.shape

        # Generate mutation mask
        should_mutate = cp.random.random((pop_size, n_params)) < mutation_rate

        # Generate mutation values
        mutation_values = cp.random.normal(0, mutation_strength, (pop_size, n_params))

        # Apply mutations
        for i in range(n_params):
            low, high = self.bounds[i]
            if low == high:
                continue

            # For non-zero values, use proportional mutation
            non_zero_mask = (gpu_pop[:, i] != 0) & should_mutate[:, i]
            if cp.any(non_zero_mask):
                gpu_pop[non_zero_mask, i] *= 1 + mutation_values[non_zero_mask, i]

            # For zero values, use absolute mutation based on bounds
            zero_mask = (gpu_pop[:, i] == 0) & should_mutate[:, i]
            if cp.any(zero_mask):
                range_size = high - low
                gpu_pop[zero_mask, i] += range_size * mutation_values[zero_mask, i]

            # Clip to bounds
            gpu_pop[:, i] = cp.clip(gpu_pop[:, i], low, high)

        # Round to significant digits
        if self.sig_digits > 0:
            # This would be a custom CUDA kernel in a real implementation
            new_population_cpu = gpu_pop.get()
            for i in range(pop_size):
                for j in range(n_params):
                    if self.bounds[j][0] != self.bounds[j][1]:
                        new_population_cpu[i, j] = round_floats(
                            new_population_cpu[i, j], self.sig_digits
                        )
            gpu_pop = cp.array(new_population_cpu)

        return gpu_pop.get()

    def selection_gpu(self, population, fitness_values, selection_pressure=0.7):
        """Perform tournament selection on GPU"""
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU acceleration requested but not available")

        pop_size = len(population)
        tournament_size = max(2, int(pop_size * selection_pressure))

        # Convert fitness to GPU array
        gpu_fitness = cp.array(fitness_values)

        # Generate tournament participants
        selected_indices = []

        for _ in range(pop_size):
            # Select random participants
            participants = cp.random.choice(pop_size, tournament_size, replace=False)

            # Get fitness values for participants
            participant_fitness = gpu_fitness[participants]

            # Find winner (minimum fitness for minimization problem)
            winner_idx = participants[cp.argmin(participant_fitness)]
            selected_indices.append(winner_idx)

        # Create new population from selected individuals
        selected_population = [population[idx] for idx in selected_indices]

        return selected_population


async def main():
    """Main entry point"""
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
        "--max-memory",
        type=int,
        default=80,
        help="Maximum memory usage percentage (client mode only)",
    )
    parser.add_argument(
        "--max-gpu",
        type=int,
        default=80,
        help="Maximum GPU memory usage percentage (client mode only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto, client mode only)",
    )

    # GPU-specific arguments
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use")

    args = parser.parse_args()

    # Set GPU device if specified
    if args.use_gpu and GPU_AVAILABLE:
        try:
            cp.cuda.Device(args.gpu_id).use()
            logging.info(f"Using GPU device {args.gpu_id}")
        except Exception as e:
            logging.error(f"Failed to set GPU device: {e}")
            args.use_gpu = False

    # Create and run the appropriate component
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
            await client.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        if args.mode == "client" and "client" in locals():
            client.cleanup()

        logging.info("Exiting...")


if __name__ == "__main__":
    # Import tempfile here to avoid circular imports
    import tempfile

    asyncio.run(main())
