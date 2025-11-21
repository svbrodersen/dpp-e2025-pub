#!/usr/bin/env python3
import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib
# Set backend to Agg to handle headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments. Only requires the two file paths."""
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSON files automatically.")
    parser.add_argument("baseline_json", type=Path, help="Path to the Sequential/C JSON file")
    parser.add_argument("accelerated_json", type=Path, help="Path to the OpenCL/CUDA JSON file")
    parser.add_argument("output", type=Path, help="Output filename")
    return parser.parse_args()

def load_json(filepath: Path) -> Dict[str, Any]:
    """Safely loads a JSON file."""
    try:
        with filepath.open('r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}.")
        sys.exit(1)

def extract_benchmark_data(data: Dict[str, Any]) -> Tuple[str, Dict[int, float]]:
    """
    Parses the JSON content dynamically.
    
    Returns:
        - benchmark_name (str): The name found in the top-level key.
        - results (Dict[int, float]): A dictionary mapping {input_size: mean_runtime_ms}
    """
    # 1. Get the top-level key (The Program/Benchmark Name)
    # We assume the file contains one main benchmark entry.
    if not data:
        print("Error: JSON file is empty.")
        sys.exit(1)
    
    program_key = next(iter(data)) # e.g., "my_program.fut:benchmark1"
    
    # Clean up the name for the plot title
    # Removes the .fut extension part for cleaner reading
    benchmark_name = program_key.split(':')[-1] 

    # 2. Extract Datasets
    try:
        datasets = data[program_key]['datasets']
    except KeyError:
        print(f"Error: structure of JSON unexpected. Could not find 'datasets' under '{program_key}'")
        sys.exit(1)

    results = {}

    # 3. Iterate through datasets and parse sizes using Regex
    # Pattern looks for digits inside square brackets: [1024]
    size_pattern = re.compile(r'\[(\d+)\]')

    for dataset_key, dataset_val in datasets.items():
        # Extract size
        match = size_pattern.search(dataset_key)
        if match:
            size = int(match.group(1))
            
            # Extract runtimes
            if 'runtimes' in dataset_val:
                runtimes = dataset_val['runtimes']
                # Calculate mean and convert microseconds to milliseconds
                avg_runtime_ms = np.mean(runtimes) / 1000.0
                results[size] = avg_runtime_ms
            else:
                print(f"Warning: No runtimes found for dataset {dataset_key}")
        else:
            print(f"Warning: Could not parse size from key '{dataset_key}'. Skipping.")

    return benchmark_name, results

def align_data(baseline_data: Dict[int, float], accel_data: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds common input sizes between the two datasets and returns sorted arrays.
    """
    # Find intersection of sizes (keys)
    common_sizes = sorted(list(set(baseline_data.keys()) & set(accel_data.keys())))
    
    if not common_sizes:
        print("Error: No common input sizes found between the two JSON files.")
        sys.exit(1)

    sizes = []
    t_baseline = []
    t_accel = []

    for n in common_sizes:
        sizes.append(n)
        t_baseline.append(baseline_data[n])
        t_accel.append(accel_data[n])

    return np.array(sizes), np.array(t_baseline), np.array(t_accel)

def create_plot(sizes: np.ndarray, 
                t_accel: np.ndarray, 
                t_base: np.ndarray, 
                speedups: np.ndarray, 
                benchmark_name: str,
                output_file: str):
    """Generates and saves the matplotlib figure."""
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: Runtimes (Left Axis) ---
    p1 = ax1.plot(sizes, t_accel, 'b-o', label='Accelerated (OpenCL)')
    p2 = ax1.plot(sizes, t_base, 'g-s', label='Baseline (Seq)')
    
    ax1.set_xlabel('Input size')
    ax1.set_ylabel('Runtime (ms)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    
    # Set x-ticks
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(sizes, rotation='vertical')
    
    # Log scales
    ax1.set_xscale('log') 
    ax1.set_yscale('log')

    # --- Plot 2: Speedup (Right Axis) ---
    ax2 = ax1.twinx()
    p3 = ax2.plot(sizes, speedups, 'k-x', label='Speedup')
    
    ax2.set_ylabel('Speedup (x)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # --- Legend & Layout ---
    plots = p1 + p2 + p3
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc='best')

    ax1.set_title(f'Benchmark: {benchmark_name}')
    fig.tight_layout()

    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

def main():
    try:
        args = parse_arguments()

        # 1. Load Data base_json = load_json(args.baseline_json)
        accel_json = load_json(args.accelerated_json)

        # 2. Extract Info (Name and {size: runtime} dicts)
        # We grab the benchmark name from the baseline file, assuming they are the same benchmark
        bench_name, base_results = extract_benchmark_data(base_json)
        _, accel_results = extract_benchmark_data(accel_json)

        print(f"Processing data for: {bench_name}")

        # 3. Align Data (Ensure we only plot sizes that exist in BOTH files)
        sizes, t_base, t_accel = align_data(base_results, accel_results)

        # 4. Calculate Speedup
        # speedup = Baseline / Accelerated
        speedups = t_base / t_accel

        # 5. Determine output filename
        out_name = args.output

        # 6. Plot
        create_plot(
            sizes=sizes,
            t_accel=t_accel,
            t_base=t_base,
            speedups=speedups,
            benchmark_name=bench_name,
            output_file=out_name
        )
    except Exception as e:
        print("error: ", e)

if __name__ == '__main__':
    main()
