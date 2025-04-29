# Passivbot Distributed Optimizer (Folding@Home Style)

This system lets you use all your available CPUs/GPUs across your network to optimize Passivbot configs, **without impacting your daily work**. It automatically adapts to your system load, scaling up when idle and backing off when busy.

## Features

- Distributed: Run clients on any number of machines (Linux, Mac, Windows, etc).
- Adaptive: Dynamically uses all available CPU/GPU, but reduces usage if you start using your machine.
- Heterogeneous: Mix and match CPUs and GPUs.
- Easy: Just run a single command on each machine.

## How to Use

### 1. Start the Server

On your main machine (with your config):

```bash
python3 src/distributed_optimize.py --mode server --config my_config.json --port 5555
```

Or for GPU:

```bash
python3 src/distributed_optimize_gpu.py --mode server --config my_config.json --port 5555
```

### 2. Start Clients

On each worker machine (Xeon, MacBook, homelab, etc):

**CPU:**
```bash
python3 src/distributed_optimize.py --mode client --server 192.168.1.100:5555
```

**GPU:**
```bash
python3 src/distributed_optimize_gpu.py --mode client --server 192.168.1.100:5555 --use-gpu
```

- The client will automatically use all available resources, but will back off if you start using your computer.
- You can run as many clients as you want.

### 3. Monitor

- The server will log progress and show connected clients.
- Results are saved in the `optimize_results/` directory.

## Advanced

- You can set `--max-cpu`, `--max-memory`, `--max-gpu` to control aggressiveness.
- The system will auto-scale worker count for best performance.

## Requirements

- Python 3.8+
- `psutil`, `numpy`, `zmq`, `cupy` (for GPU), `numba` (for GPU)
- Rust (for Passivbot backtesting core)

## Notes

- On Linux, the client process runs at lower priority by default.
- On Mac/Windows, you may need to adjust process priority manually for best experience.