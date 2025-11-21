#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import matplotlib
# Set backend to Agg before importing pyplot to handle headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSON files.")
    parser.add_argument("progname", type=str, help="Name of the program (e.g., 'mybench')")
    parser.add_argument("benchmark", type=str, help="Name of the specific benchmark")
    parser.add_argument("data_sizes", type=int, nargs='+', help="List of input data sizes (integers)")
    return parser.parse_args()

def load_json(filepath: Path) -> Dict[str, Any]:
    """Safely loads a JSON file."""
    try:
        with filepath.open('r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)

def get_runtimes(data: Dict[str, Any], program_key: str, sizes: List[int]) -> np.ndarray:
    """
    Extracts runtimes for specific data sizes from the JSON structure.
    Returns a numpy array of runtimes in milliseconds.
    """
    try:
        datasets = data[program_key]['datasets']
        runtimes = []
        
        for n in sizes:
            # This key format seems specific to your use case (Futhark benchmarks?)
            # Adjust the format string below if the JSON structure changes.
            dataset_key = f'[{n}]i32 [{n}]i32' 
            
            measurements = datasets[dataset_key]['runtimes']
            # Calculate mean and convert to milliseconds (assuming input is microseconds)
            avg_runtime_ms = np.mean(measurements) / 1000.0
            runtimes.append(avg_runtime_ms)
            
        return np.array(runtimes)

    except KeyError as e:
        print(f"Error: Key {e} not found in JSON data. Check your program name or data sizes.")
        sys.exit(1)

def create_plot(sizes: List[int], 
                opencl_times: np.ndarray, 
                c_times: np.ndarray, 
                speedups: np.ndarray, 
                benchmark_name: str,
                output_file: str):
    """Generates and saves the matplotlib figure."""
    
    fig, ax1 = plt.subplots(figsize=(10, 6)) # Added figsize for better default proportions

    # --- Plot 1: Runtimes (Left Axis) ---
    # Use explicit markers (o, s) to make data points visible on lines
    p1 = ax1.plot(sizes, opencl_times, 'b-o', label='OpenCL runtime')
    p2 = ax1.plot(sizes, c_times, 'g-s', label='Sequential runtime')
    
    ax1.set_xlabel('Input size')
    ax1.set_ylabel('Runtime (ms)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    
    # Set x-ticks to match the specific data sizes provided
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(sizes, rotation='vertical')
    
    # Log scales often look better for benchmarks, assuming sizes grow exponentially
    ax1.set_xscale('log') 
    ax1.set_yscale('log')

    # --- Plot 2: Speedup (Right Axis) ---
    ax2 = ax1.twinx()
    p3 = ax2.plot(sizes, speedups, 'k-x', label='OpenCL speedup')
    
    ax2.set_ylabel('Speedup (x)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    # Usually speedup is linear, but uncomment below if speedups vary wildly
    # ax2.set_yscale('log') 

    # --- Legend & Layout ---
    # Combine legends from both axes
    plots = p1 + p2 + p3
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc='best') # 'best' automatically finds the least crowded spot

    ax1.set_title(f'Benchmark: {benchmark_name}')
    fig.tight_layout()

    # Save
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig) # Close memory

def main():
    args = parse_arguments()

    # File definitions
    opencl_path = Path(f'{args.progname}-opencl.json')
    c_path = Path(f'{args.progname}-c.json')
    
    # Load Data
    opencl_data = load_json(opencl_path)
    c_data = load_json(c_path)

    # Helper key
    prog_key = f'{args.progname}.fut:{args.benchmark}'

    # Extract Data
    t_opencl = get_runtimes(opencl_data, prog_key, args.data_sizes)
    t_c = get_runtimes(c_data, prog_key, args.data_sizes)

    # Calculate Speedup (Vectorized division)
    # Handle division by zero if necessary (using np.where or similar)
    speedups = t_c / t_opencl

    # Plot
    create_plot(
        sizes=args.data_sizes,
        opencl_times=t_opencl,
        c_times=t_c,
        speedups=speedups,
        benchmark_name=args.benchmark,
        output_file=f'{args.benchmark}.pdf'
    )

if __name__ == '__main__':
    main()
