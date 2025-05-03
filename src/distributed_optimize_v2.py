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
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

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

        # ZMQ sockets
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.pub_socket = self.context.socket(zmq.PUB)

        # Task generation parameters - increased batch size
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

        # If client was previously marked as disconnected but is now sending messages,
        # re-register it automatically
        if client_id not in self.clients and msg_type != "register":
            logging.info(f"Client {client_id} reconnected, requesting re-registration")
            await self.router_socket.send_multipart(
                [
                    client_id.encode() if isinstance(client_id, str) else client_id,
                    json.dumps({"type": "reregister"}).encode(),
                ]
            )
            return

        if msg_type == "register":
            # New client registration
            hostname = message.get("hostname", "unknown")
            resource_info = message.get("resources", {})
            self.clients[client_id] = {
                "hostname": hostname,
                "resources": resource_info,
                "last_seen": time.time(),
                "status": "idle",
                "tasks_completed": 0,
            }
            logging.info(f"New client registered: {hostname} ({client_id})")
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
        """Send optimization tasks to a client with adaptive batch sizing"""
        if not self.population:
            # No more tasks to send
            return

        # Ensure client_id is properly decoded
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8")

        # Get client info
        client_info = self.clients.get(client_id)
        if not client_info:
            return

        # Determine batch size based on client resources
        client_cpu_count = client_info.get("resources", {}).get("cpu_count", 4)
        client_workers = client_info.get("resources", {}).get("workers", 2)

        # Scale batch size based on client's worker count
        # More powerful clients get larger batches
        adaptive_batch_size = min(
            max(
                self.batch_size, client_workers * 3
            ),  # At least 3 individuals per worker
            len(self.population),
        )

        # Get batch of individuals to evaluate
        individuals = self.population[:adaptive_batch_size]
        self.population = self.population[adaptive_batch_size:]

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
            f"Sent task {task_id} with {len(individuals)} individuals to client {client_id} (workers: {client_workers})"
        )

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

        for client_id, client_info in list(
            self.clients.items()
        ):  # Use list() to avoid modification during iteration
            # Check if client hasn't sent a heartbeat in 10 minutes (increased from 60 seconds)
            if current_time - client_info["last_seen"] > 600:  # 10 minutes
                logging.warning(
                    f"Client {client_id} ({client_info['hostname']}) appears to be disconnected"
                )
                disconnected_clients.append(client_id)

                # If client had pending tasks, return them to the queue
                for task_id, individuals in list(self.pending_tasks.items()):
                    if (
                        task_id in self.task_progress
                        and self.task_progress[task_id]["client_id"] == client_id
                    ):
                        logging.info(f"Returning task {task_id} to queue")
                        self.population.extend(individuals)
                        self.pending_tasks.pop(task_id)

                        # Remove from task progress
                        if task_id in self.task_progress:
                            del self.task_progress[task_id]

        # Remove disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:  # Check again to avoid KeyError
                self.clients.pop(client_id)

    async def monitor_tasks(self):
        """Monitor tasks and restart any that appear stuck"""
        # Dictionary to track when tasks were created
        task_creation_times = {}

        while True:
            current_time = time.time()
            stuck_tasks = []

            # Check for stuck tasks (no progress update in 10 minutes)
            for task_id, individuals in list(self.pending_tasks.items()):
                # Record creation time if not already tracked
                if task_id not in task_creation_times:
                    task_creation_times[task_id] = current_time

                # Get task progress if available
                task_progress = self.task_progress.get(task_id)

                # Check if task is stuck
                if task_progress:
                    last_update = task_progress.get("last_update", 0)
                    if current_time - last_update > 600:  # 10 minutes
                        stuck_tasks.append((task_id, individuals))
                else:
                    # No progress info - check if task is old (created more than 15 minutes ago)
                    # This is a fallback for tasks that never reported progress
                    creation_time = task_creation_times.get(task_id, current_time)
                    if current_time - creation_time > 900:  # 15 minutes
                        stuck_tasks.append((task_id, individuals))

            # Restart stuck tasks
            for task_id, individuals in stuck_tasks:
                logging.warning(
                    f"Task {task_id} appears to be stuck - returning to queue"
                )
                self.pending_tasks.pop(task_id)
                self.population.extend(individuals)

                # Remove from task progress
                if task_id in self.task_progress:
                    del self.task_progress[task_id]

                # Remove from creation times tracking
                if task_id in task_creation_times:
                    del task_creation_times[task_id]

            # Clean up task_creation_times for completed tasks
            for task_id in list(task_creation_times.keys()):
                if task_id not in self.pending_tasks:
                    del task_creation_times[task_id]

            await asyncio.sleep(60)  # Check every minute

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

        logging.info(
            f"Starting with {self.n_workers} workers (out of {total_cpus} CPUs)"
        )

        # Create process pool for CPU-bound tasks
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)

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
        # Create a simple CPU-bound task

    def prewarm_process_pool(self):
        """
        Prewarm the process pool by submitting dummy tasks.
        This creates the worker processes upfront so they're ready when real tasks arrive.
        """
        logging.info(f"Prewarming process pool with {self.n_workers} workers...")

        # Submit one task per worker
        futures = []
        for i in range(self.n_workers):
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

    async def setup(self):
        """Connect to server and register"""
        # Set client ID in socket
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.node_id.encode())

        # Connect to server
        server_host, server_port = self.server_address.split(":")
        self.dealer_socket.connect(f"tcp://{server_host}:{server_port}")
        self.sub_socket.connect(f"tcp://{server_host}:{int(server_port)+1}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        # Register with server
        success = await self.register_with_server()

        if not success:
            sys.exit(1)

    async def initialize_evaluator(self):
        """Initialize the optimization evaluator"""
        logging.info("Initializing evaluator...")

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

    async def process_task(self, task):
        """Process an optimization task using process pool"""
        task_id = task["task_id"]
        individuals = task["individuals"]
        overrides_list = self.config.get("optimize", {}).get("enable_overrides", [])

        logging.info(f"Processing task {task_id} with {len(individuals)} individuals")
        logging.info(f"Using {self.n_workers} worker processes")

        results = []
        valid_results = 0
        total_individuals = len(individuals)
        processed = 0

        # Process individuals with resource management
        futures = []

        # Submit all individuals to the process pool
        for individual in individuals:
            # Check if we should pause due to high resource usage
            await self.resource_manager.wait_for_resources()

            # Submit the task to the process pool
            future = self.process_pool.submit(
                evaluate_individual_wrapper,
                individual,
                overrides_list,
                self.config,
                self.shared_memory_files,
                self.hlcvs_shapes,
                self.hlcvs_dtypes,
                self.btc_usd_shared_memory_files,
                self.btc_usd_dtypes,
                self.msss,
            )
            futures.append(future)

        # Process results as they complete
        start_time = time.time()
        last_progress_time = start_time

        for future in futures:
            try:
                # Wait for the future to complete
                result = future.result()

                if result:
                    results.append(result)
                    valid_results += 1

                # Update progress
                processed += 1
                current_time = time.time()

                # Send progress update every 5 seconds or every 10% of individuals
                if (
                    current_time - last_progress_time > 5
                    or processed % max(1, total_individuals // 10) == 0
                ):
                    await self.dealer_socket.send(
                        json.dumps(
                            {
                                "type": "progress",
                                "task_id": task_id,
                                "processed": processed,
                                "total": total_individuals,
                                "valid_results": valid_results,
                            }
                        ).encode()
                    )
                    last_progress_time = current_time

                # Periodically optimize memory if enabled
                if processed % 10 == 0:
                    self.optimize_memory_usage()

            except Exception as e:
                logging.error(f"Error processing individual: {e}")
                processed += 1  # Still count as processed even if it failed

        end_time = time.time()
        processing_time = end_time - start_time

        logging.info(
            f"Processed {processed}/{total_individuals} individuals in {processing_time:.2f} seconds"
        )
        logging.info(
            f"Average time per individual: {processing_time/total_individuals:.2f} seconds"
        )
        logging.info(f"Valid results: {valid_results}/{total_individuals}")

        # Send final progress update
        await self.dealer_socket.send(
            json.dumps(
                {
                    "type": "progress",
                    "task_id": task_id,
                    "processed": processed,
                    "total": total_individuals,
                    "valid_results": valid_results,
                }
            ).encode()
        )

        # Send results back to server
        await self.dealer_socket.send(
            json.dumps(
                {"type": "result", "task_id": task_id, "results": results}
            ).encode()
        )

        # Signal ready for more tasks immediately
        await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())

        logging.info(f"Completed task {task_id} with {len(results)} valid results")

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

                await self.dealer_socket.send(
                    json.dumps(
                        {"type": "heartbeat", "resources": resource_info}
                    ).encode()
                )

                # Wait for acknowledgment with a timeout
                try:
                    # Set a timeout for receiving the acknowledgment
                    response = await asyncio.wait_for(
                        self.dealer_socket.recv(), timeout=5.0
                    )
                    message = json.loads(response.decode())

                    # If server requests re-registration, do it
                    if message.get("type") == "reregister":
                        logging.info("Server requested re-registration")
                        await self.register_with_server()
                except asyncio.TimeoutError:
                    logging.warning(
                        "No heartbeat acknowledgment received from server, will retry"
                    )

            except Exception as e:
                logging.error(f"Error sending heartbeat: {e}")

            # Send heartbeat every 15 seconds
            await asyncio.sleep(15)  # Send heartbeat every 15 seconds

    async def register_with_server(self):
        """Register with the server"""
        resource_info = {
            "cpu_count": multiprocessing.cpu_count(),
            "workers": self.n_workers,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
            "total_memory": psutil.virtual_memory().total,
        }

        await self.dealer_socket.send(
            json.dumps(
                {
                    "type": "register",
                    "hostname": self.hostname,
                    "resources": resource_info,
                }
            ).encode()
        )

        # Wait for registration confirmation
        message = await self.dealer_socket.recv()
        message = json.loads(message.decode())

        if message["type"] == "registered":
            logging.info(f"Successfully registered with server")
            # Only update config if it's provided and we don't already have one
            if "config" in message and self.config is None:
                self.config = message["config"]
                # Initialize evaluator if needed
                if self.evaluator is None:
                    await self.initialize_evaluator()

            # Signal ready for tasks
            await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())
            return True
        else:
            logging.error(f"Failed to register with server: {message}")
            return False

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
        """Dynamically adjust worker count based on system load (CPU and memory)."""
        min_workers = 1
        max_workers = (
            multiprocessing.cpu_count()
        )  # Allow using all CPUs if system is idle

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

                # If worker count changed, log it and update the process pool
                if self.n_workers != old_workers:
                    logging.info(
                        f"Adjusting workers: {old_workers} → {self.n_workers} "
                        + f"(CPU: {avg_cpu:.1f}%, Mem: {avg_mem:.1f}%)"
                    )
                    self.last_worker_adjustment = current_time

                    # Recreate the process pool with the new worker count
                    # This is a bit expensive but ensures we adapt to changing conditions
                    try:
                        old_pool = self.process_pool
                        self.process_pool = ProcessPoolExecutor(
                            max_workers=self.n_workers
                        )

                        # Shutdown old pool gracefully
                        old_pool.shutdown(wait=False)

                        # Prewarm the new pool
                        self.prewarm_process_pool()
                    except Exception as e:
                        logging.error(f"Error recreating process pool: {e}")

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

            # If user is active, temporarily reduce worker count
            if user_active and self.n_workers > min_workers:
                temp_workers = max(min_workers, int(self.n_workers * 0.5))
                if temp_workers != self.n_workers:
                    logging.info(
                        f"User activity detected - temporarily reducing workers to {temp_workers}"
                    )
                    self.n_workers = temp_workers

            await asyncio.sleep(2)

    async def run(self):
        """Main client loop"""
        # Start heartbeat immediately
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        await self.setup()
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
                elif message["type"] == "reregister":
                    # Server doesn't recognize us, re-register
                    logging.info("Server requested re-registration")
                    resource_info = {
                        "cpu_count": multiprocessing.cpu_count(),
                        "workers": self.n_workers,
                        "max_cpu_percent": self.max_cpu_percent,
                        "max_memory_percent": self.max_memory_percent,
                        "total_memory": psutil.virtual_memory().total,
                    }
                    await self.dealer_socket.send(
                        json.dumps(
                            {
                                "type": "register",
                                "hostname": self.hostname,
                                "resources": resource_info,
                            }
                        ).encode()
                    )
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
        # Shutdown process pool
        if hasattr(self, "process_pool"):
            logging.info("Shutting down process pool...")
            self.process_pool.shutdown(wait=True)

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


async def main():
    """Main entry point"""
    # Ensure Rust code is compiled
    manage_rust_compilation()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Distributed optimization for Passivbot with improved CPU utilization"
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
        default=25,  # Increased from 10 to 25
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
        default=85,  # Increased from 70 to 85
        help="Maximum CPU usage percentage (client mode only)",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=85,  # Increased from 80 to 85
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
        choices=["low", "normal", "high"],
        default="normal",
        help="Process priority (client mode only)",
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Periodically optimize memory usage (client mode only)",
    )

    args = parser.parse_args()

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
    asyncio.run(main())
