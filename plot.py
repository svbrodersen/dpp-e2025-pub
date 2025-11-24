#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import matplotlib
# Set backend to Agg before importing pyplot to handle headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data Structure for Extracted Results ---
# {
#     'benchmark_name': {
#         'sizes': np.ndarray[int],
#         'opencl_runtimes': np.ndarray[float],
#         'c_runtimes': np.ndarray[float],
#         'speedups': np.ndarray[float],
#     }
# }
BenchmarkResults = Dict[str, Dict[str, Any]]

# --- Utility Functions ---

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Plot all benchmark results from JSON files.")
    parser.add_argument("progname", type=str, help="Base name of the program (e.g., 'mybench'). Files are expected to be <progname>-opencl.json and <progname>-c.json.")
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

# --- Core Data Extraction Logic ---

def extract_all_benchmark_data(opencl_data: Dict[str, Any], c_data: Dict[str, Any]) -> BenchmarkResults:
    """
    Parses both JSON datasets to extract runtimes and calculate speedups
    for ALL found benchmarks and data sizes.
    """
    all_results: BenchmarkResults = {}
    
    # The structure key is '<program_name>.fut:<benchmark>'
    for full_key, c_data_item in c_data.items():
        if full_key not in opencl_data:
            print(f"Warning: Key '{full_key}' found in C data but not OpenCL data. Skipping.")
            continue
        
        # 1. Extract benchmark name
        try:
            benchmark_name = full_key.split(':')[-1]
        except IndexError:
            print(f"Warning: Unexpected key format '{full_key}'. Skipping.")
            continue

        # 2. Extract sizes, C runtimes, and OpenCL runtimes
        opencl_datasets = opencl_data[full_key]['datasets']
        c_datasets = c_data_item['datasets']

        sizes: List[int] = []
        opencl_runtimes: List[float] = []
        c_runtimes: List[float] = []

        # Iterate over all dataset keys (e.g., '[1024]i32 [1024]i32')
        for dataset_key, c_dataset in c_datasets.items():
            if dataset_key not in opencl_datasets:
                print(f"Warning: Dataset key '{dataset_key}' for benchmark '{benchmark_name}' missing in OpenCL data. Skipping.")
                continue

            opencl_dataset = opencl_datasets[dataset_key]
            
            # Extract size N from the dataset_key
            try:
                size_str = dataset_key.split(']')[0].split('[')[1]
                size_n = int(size_str)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse size from dataset key '{dataset_key}'. Skipping.")
                continue
            
            try:
                # Calculate mean runtime in milliseconds
                avg_c_ms = np.mean(c_dataset['runtimes']) / 1000.0
                avg_opencl_ms = np.mean(opencl_dataset['runtimes']) / 1000.0
            except KeyError as e:
                print(f"Error extracting runtimes for '{benchmark_name}' at size {size_n}: Key {e} not found. Skipping.")
                continue

            sizes.append(size_n)
            c_runtimes.append(avg_c_ms)
            opencl_runtimes.append(avg_opencl_ms)
            
        # 3. Process collected data
        if not sizes:
            continue
            
        # Sort data by size
        sorted_indices = np.argsort(sizes)
        
        np_sizes = np.array(sizes)[sorted_indices]
        np_c_times = np.array(c_runtimes)[sorted_indices]
        np_opencl_times = np.array(opencl_runtimes)[sorted_indices]
        np_speedups = np_c_times / np_opencl_times

        # 4. Store results
        all_results[benchmark_name] = {
            'sizes': np_sizes,
            'opencl_runtimes': np_opencl_times,
            'c_runtimes': np_c_times,
            'speedups': np_speedups,
        }

    return all_results

# --- Plotting Functions ---

def create_plot(sizes: np.ndarray,
                opencl_times: np.ndarray,
                c_times: np.ndarray,
                speedups: np.ndarray,
                benchmark_name: str,
                output_file: str):
    """Generates and saves the individual matplotlib figure."""
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: Runtimes (Left Axis) ---
    p1 = ax1.plot(sizes, opencl_times, 'b-o', label='OpenCL runtime')
    p2 = ax1.plot(sizes, c_times, 'g-s', label='Sequential runtime')
    
    ax1.set_xlabel('Input size')
    ax1.set_ylabel('Runtime (ms)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    
    # Set x-ticks to match the specific data sizes provided
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(sizes, rotation='vertical')
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)

    # --- Plot 2: Speedup (Right Axis) ---
    ax2 = ax1.twinx()
    p3 = ax2.plot(sizes, speedups, 'k-x', label='OpenCL speedup')
    
    ax2.set_ylabel('Speedup (x)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # --- Legend & Layout ---
    plots = p1 + p2 + p3
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc='best')

    ax1.set_title(f'Benchmark: {benchmark_name}')
    fig.tight_layout()

    print(f"Saving plot for {benchmark_name} to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

def create_combined_metric_plot(all_results: BenchmarkResults, 
                                metric_key: str, 
                                output_file: str):
    """
    Generates a single plot containing lines for ALL benchmarks 
    for a specific metric (e.g., 'opencl_runtimes' or 'c_runtimes').
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Iterate through all benchmarks and plot them on the same axis
    for benchmark_name, data in all_results.items():
        if data['sizes'].size > 0:
            ax.plot(data['sizes'], 
                    data[metric_key], 
                    marker='o', 
                    markersize=4, 
                    label=benchmark_name)

    ax.set_xlabel('Input size')
    ax.set_ylabel('Runtime (ms)')
    
    # Clean title based on key
    title_str = "OpenCL" if "opencl" in metric_key else "Sequential C"
    ax.set_title(f'Combined {title_str} Runtimes - All Benchmarks')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    fig.tight_layout()
    
    print(f"Saving combined plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

# --- Main Execution ---

def main():
    args = parse_arguments()

    # 1. File definitions
    opencl_path = Path(f'{args.progname}-opencl.json')
    c_path = Path(f'{args.progname}-c.json')
    
    # 2. Load Data
    opencl_data = load_json(opencl_path)
    c_data = load_json(c_path)

    # 3. Extract Results
    all_results = extract_all_benchmark_data(opencl_data, c_data)
    
    if not all_results:
        print("No complete benchmark data found to plot. Exiting.")
        sys.exit(0)

    # 4. Plot Individual Results
    for benchmark_name, data in all_results.items():
        if data['sizes'].size > 0:
            create_plot(
                sizes=data['sizes'],
                opencl_times=data['opencl_runtimes'],
                c_times=data['c_runtimes'],
                speedups=data['speedups'],
                benchmark_name=benchmark_name,
                output_file=f'{benchmark_name}.pdf'
            )
        else:
            print(f"Warning: No valid data points to plot for benchmark '{benchmark_name}'.")

    # 5. Plot Combined Results (New Functionality)
    print("Generating combined plots...")
    create_combined_metric_plot(all_results, 'opencl_runtimes', 'combined-opencl.pdf')
    create_combined_metric_plot(all_results, 'c_runtimes', 'combined-c.pdf')

if __name__ == '__main__':
    main()
