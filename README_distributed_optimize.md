# Distributed Optimization for Passivbot

This tool allows you to distribute Passivbot optimization workloads across multiple machines.

## Requirements

- Python 3.8+
- ZeroMQ (`pip install pyzmq`)
- All regular Passivbot dependencies

## Architecture

The system uses a client-server architecture:

1. **Server**: Coordinates optimization tasks and maintains the population and Pareto front
2. **Clients**: Connect to the server, receive optimization tasks, and return results

## Usage

### Server

Start the server on a machine that will coordinate the optimization:

```bash
python distributed_optimize.py --mode server --port 5555 --config configs/my_config.json --batch-size 10
```

Options:
- `--port`: Port to listen on (default: 5555)
- `--config`: Path to Passivbot configuration file
- `--batch-size`: Number of individuals to send in each task (default: 10)

### Client

Start clients on each machine that will perform optimization tasks:

```bash
python distributed_optimize.py --mode client --server 192.168.1.100:5555
```

Options:
- `--server`: Server address in format host:port

## How It Works

1. The server loads the configuration and initializes the optimization population
2. Clients connect to the server and download the configuration
3. Clients download and prepare market data
4. The server distributes optimization tasks to available clients
5. Clients evaluate individuals and return results to the server
6. The server updates the Pareto front and generates new individuals
7. Results are continuously saved to the results directory

## Tips

- The server should be run on a machine with good network connectivity
- Clients can be added or removed at any time
- If a client disconnects during a task, the server will reassign the task
- Results are saved in the `optimize_results` directory on the server

## Monitoring

The server broadcasts status updates that clients display in their logs. You can also check the `server_status.json` file in the results directory for current status.
```

## How to Use the Distributed Optimizer

1. Copy the `distributed_optimize.py` script to your Passivbot directory
2. Start the server on your main machine:
   ```bash
   python distributed_optimize.py --mode server --config configs/your_config.json
   ```
3. Start clients on other machines:
   ```bash
   python distributed_optimize.py --mode client --server 192.168.1.100:5555