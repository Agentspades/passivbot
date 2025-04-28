# GPU-Accelerated Distributed Optimization for Passivbot

This tool allows you to distribute Passivbot optimization workloads across multiple machines with GPU acceleration for faster performance.

## Requirements

- Python 3.8+
- ZeroMQ (`pip install pyzmq`)
- For GPU acceleration:
  - CUDA-compatible GPU
  - CuPy (`pip install cupy-cuda11x` - replace with your CUDA version)
  - Numba (`pip install numba`)
- All regular Passivbot dependencies

## Architecture

The system uses a client-server architecture with GPU acceleration:

1. **Server**: Coordinates optimization tasks and maintains the population and Pareto front
2. **Clients**: Connect to the server, receive optimization tasks, and return results
3. **GPU Acceleration**: Clients with GPUs can accelerate:
   - Parallel evaluation of multiple individuals
   - Genetic algorithm operations (crossover, mutation)
   - Backtesting calculations

## Usage

### Server

Start the server on a machine that will coordinate the optimization:

```bash
python distributed_optimize_gpu.py --mode server --port 5555 --config configs/my_config.json --batch-size 32
```

Options:
- `--port`: Port to listen on (default: 5555)
- `--config`: Path to Passivbot configuration file
- `--batch-size`: Number of individuals to send in each task (default: 10)

### Client

Start clients on each machine that will perform optimization tasks:

```bash
python distributed_optimize_gpu.py --mode client --server 192.168.1.100:5555 --use-gpu --gpu-id 0
```

Options:
- `--server`: Server address in format host:port
- `--use-gpu`: Enable GPU acceleration if available
- `--gpu-id`: Specify which GPU to use (default: 0)

## GPU Acceleration

The tool automatically detects if CUDA-compatible GPUs are available and uses them for:

1. **Batch Evaluation**: Evaluating multiple individuals simultaneously
2. **Genetic Operations**: Performing crossover and mutation on the GPU
3. **Backtesting Calculations**: Accelerating the most computationally intensive parts of backtesting

If a GPU is not available or if GPU operations fail, the system automatically falls back to CPU processing.

## Performance Considerations

- **Memory Usage**: GPU acceleration requires transferring market data to GPU memory, which may be limited
- **Batch Size**: Larger batch sizes generally improve GPU utilization but require more memory
- **Multiple GPUs**: You can run multiple clients on a machine with multiple GPUs by specifying different `--gpu-id` values

## Monitoring

The server broadcasts status updates that clients display in their logs. You can also check the `server_status.json` file in the results directory for current status.

## Tips for Best Performance

1. **Client Selection**: Use clients with powerful GPUs for best performance
2. **Batch Size Tuning**: Experiment with different batch sizes to find the optimal value for your GPU
3. **Memory Management**: Monitor GPU memory usage and adjust batch size if you encounter out-of-memory errors
4. **Network Bandwidth**: Ensure good network connectivity between server and clients to minimize latency