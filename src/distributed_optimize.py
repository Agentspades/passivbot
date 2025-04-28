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
        msg_type = message.get("type")

        if msg_type == "register":
            # New client registration
            hostname = message.get("hostname", "unknown")
            self.clients[client_id] = {
                "hostname": hostname,
                "last_seen": time.time(),
                "status": "idle",
                "tasks_completed": 0,
            }
            logging.info(f"New client registered: {hostname} ({client_id})")
            await self.router_socket.send_multipart(
                [
                    client_id,
                    json.dumps({"type": "registered", "config": self.config}).encode(),
                ]
            )

        elif msg_type == "heartbeat":
            # Update client's last seen timestamp
            if client_id in self.clients:
                self.clients[client_id]["last_seen"] = time.time()
                await self.router_socket.send_multipart(
                    [client_id, json.dumps({"type": "ack"}).encode()]
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
            [client_id, json.dumps(task_message).encode()]
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
        await self.dealer_socket.send(
            json.dumps({"type": "register", "hostname": self.hostname}).encode()
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

        logging.info("Evaluator initialization complete")

    async def process_task(self, task):
        """Process an optimization task"""
        task_id = task["task_id"]
        individuals = task["individuals"]
        overrides_list = self.config.get("optimize", {}).get("enable_overrides", [])

        logging.info(f"Processing task {task_id} with {len(individuals)} individuals")

        results = []
        for individual in individuals:
            try:
                # Evaluate individual
                objectives = self.evaluator.evaluate(individual, overrides_list)

                # Get result from queue
                result = await self.results_queue.get()

                # Add individual to result for tracking
                result["individual"] = individual

                # Add to results list
                results.append(result)

            except Exception as e:
                logging.error(f"Error evaluating individual: {e}")
                import traceback

                traceback.print_exc()

        # Send results back to server
        await self.dealer_socket.send(
            json.dumps(
                {"type": "result", "task_id": task_id, "results": results}
            ).encode()
        )

        # Signal ready for more tasks
        await self.dealer_socket.send(json.dumps({"type": "ready"}).encode())

        logging.info(f"Completed task {task_id}")

    async def heartbeat_loop(self):
        """Send periodic heartbeats to server"""
        while True:
            try:
                await self.dealer_socket.send(
                    json.dumps({"type": "heartbeat"}).encode()
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

    async def run(self):
        """Main client loop"""
        await self.setup()

        # Start background tasks
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        status_task = asyncio.create_task(self.status_listener())

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
        # Remove shared memory files
        for shared_memory_file in self.shared_memory_files.values():
            if shared_memory_file and os.path.exists(shared_memory_file):
                logging.info(f"Removing shared memory file: {shared_memory_file}")
                try:
                    os.unlink(shared_memory_file)
                except Exception as e:
                    logging.error(f"Error removing shared memory file: {e}")


async def main():
    """Main entry point"""
    # Ensure Rust code is compiled
    manage_rust_compilation()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Distributed optimization for Passivbot"
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
    # Import tempfile here to avoid circular imports
    import tempfile

    asyncio.run(main())
