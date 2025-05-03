#!/usr/bin/env python3
import multiprocessing
import time
import psutil
import argparse
import os


def cpu_intensive_task(duration=10):
    """A CPU-intensive task that runs for the specified duration"""
    start_time = time.time()
    while time.time() - start_time < duration:
        # Perform a CPU-intensive calculation
        for i in range(10000000):
            _ = i * i


def run_test(num_processes, duration):
    """Run a CPU stress test with the specified number of processes"""
    print(
        f"Starting CPU stress test with {num_processes} processes for {duration} seconds"
    )
    print(f"System has {psutil.cpu_count()} logical CPUs")

    # Create and start processes
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=cpu_intensive_task, args=(duration,))
        processes.append(p)
        p.start()

    # Monitor CPU usage during the test
    start_time = time.time()
    while (
        time.time() - start_time < duration + 1
    ):  # +1 to ensure we capture the full duration
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("CPU stress test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Stress Test")
    parser.add_argument(
        "--processes",
        type=int,
        default=psutil.cpu_count(),
        help="Number of processes to use (default: all CPUs)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration of the test in seconds (default: 10)",
    )

    args = parser.parse_args()
    run_test(args.processes, args.duration)
