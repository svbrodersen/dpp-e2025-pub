#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import re

import matplotlib
# Set backend to Agg before importing pyplot to handle headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for backends ---
@dataclass
class Backend:
    """Configuration for a single backend type."""
    name: str           # Display name (e.g., "OpenCL", "Sequential C")
    file_suffix: str    # File suffix (e.g., "opencl", "c")
    color: str          # Plot color
    marker: str         # Plot marker style
    
# Define all possible backends here - add/remove as needed
AVAILABLE_BACKENDS = [
    Backend(name="OpenCL", file_suffix="opencl", color='b', marker='o'),
    Backend(name="Sequential C", file_suffix="c", color='g', marker='s'),
    Backend(name="Multicore", file_suffix="multicore", color='r', marker='^'),
    Backend(name="CUDA", file_suffix="cuda", color='m', marker='D'),
    Backend(name="ISPC", file_suffix="ispc", color='c', marker='v'),
]

# Specify which backend to use as baseline for speedup calculations (typically the sequential one)
BASELINE_BACKEND = "c"

# --- Data Structure for Extracted Results ---
BenchmarkResults = Dict[str, Dict[str, Any]]

# --- Utility Functions ---

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot all benchmark results from JSON files."
    )

    parser.add_argument(
        "progname",
        type=str,
        help="Base name of the program (e.g., 'mybench'). Files: <progname>-<backend>.json."
    )

    parser.add_argument(
        "--xbase",
        type=int,
        default=None,
        help="Log scale base for X axis. Omit for linear scale."
    )

    parser.add_argument(
        "--ybase",
        type=int,
        default=None,
        help="Log scale base for Y axis. Omit for linear scale."
    )

    return parser.parse_args()

def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Safely loads a JSON file. Returns None if file doesn't exist."""
    try:
        with filepath.open('r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)

def parse_dataset_key(dataset_key: str) -> Optional[List[Any]]:
    """
    Parses dataset keys in the format: '#N ("val1 val2 val3 ...")'
    Returns a list of parsed values (converted to int/float where possible).
    """
    # Extract the content within quotes
    match = re.search(r'"([^"]+)"', dataset_key)
    if not match:
        return None
    
    params_str = match.group(1)
    params = params_str.split()
    
    # Parse each parameter
    parsed_params = []
    for param in params:
        # Remove type suffix (e.g., 'f32', 'i64', 'i32')
        value_str = re.sub(r'[a-z]\d+$', '', param)
        try:
            # Try to parse as int first, then float
            if '.' in value_str:
                parsed_params.append(float(value_str))
            else:
                parsed_params.append(int(value_str))
        except ValueError:
            parsed_params.append(value_str)
    
    return parsed_params

def identify_varying_parameter(datasets: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """
    Identifies which parameter index varies across datasets and what it represents.
    Returns (parameter_index, parameter_name) or (None, None) if unclear.
    """
    all_params = []
    for dataset_key in datasets.keys():
        params = parse_dataset_key(dataset_key)
        if params:
            all_params.append(params)
    
    if not all_params or len(all_params) < 2:
        return None, None
    
    # Find which parameter position varies
    num_params = len(all_params[0])
    varying_indices = []
    
    for i in range(num_params):
        values = [params[i] for params in all_params if i < len(params)]
        if len(set(values)) > 1:  # This parameter varies
            varying_indices.append(i)
    
    if len(varying_indices) == 1:
        idx = varying_indices[0]
        # Common parameter names based on position for Ising model
        # Format is typically: h, w, n (height, width, iterations)
        param_names = {2: 'h', 3: 'w', 4: 'n'}
        return idx, param_names.get(idx, f'param{idx}')
    elif len(varying_indices) > 1:
        # Multiple parameters vary - use the last one (often the most significant)
        idx = varying_indices[-1]
        param_names = {2: 'h', 3: 'w', 4: 'n'}
        return idx, param_names.get(idx, f'param{idx}')
    
    return None, None

# --- Core Data Extraction Logic ---

def extract_all_benchmark_data(backend_data: Dict[str, Dict[str, Any]], 
                                available_backends: List[str]) -> BenchmarkResults:
    """
    Parses all backend JSON datasets to extract runtimes and calculate speedups
    for ALL found benchmarks with varying parameters.
    """
    all_results: BenchmarkResults = {}
    
    # Use the first available backend as reference for benchmark names
    reference_backend = available_backends[0]
    reference_data = backend_data[reference_backend]
    
    # Iterate through all benchmark keys in reference data
    for full_key, ref_data_item in reference_data.items():
        # 1. Extract benchmark name
        try:
            benchmark_name = full_key.split(':')[-1]
        except IndexError:
            print(f"Warning: Unexpected key format '{full_key}'. Skipping.")
            continue

        # 2. Identify varying parameter
        ref_datasets = ref_data_item['datasets']
        param_idx, param_name = identify_varying_parameter(ref_datasets)
        
        if param_idx is None:
            print(f"Warning: Could not identify varying parameter for '{benchmark_name}'. Skipping.")
            continue
        
        print(f"Processing '{benchmark_name}' - varying parameter: {param_name} (index {param_idx})")
        
        # 3. Extract parameter values and runtimes for each backend
        sizes: List[float] = []
        backend_runtimes: Dict[str, List[float]] = {backend: [] for backend in available_backends}

        # Iterate over all dataset keys
        for dataset_key, ref_dataset in ref_datasets.items():
            # Parse parameters from the dataset key
            params = parse_dataset_key(dataset_key)
            if params is None or param_idx >= len(params):
                print(f"Warning: Could not parse parameters from '{dataset_key}'. Skipping.")
                continue
            
            param_value = params[param_idx]
            
            # Check if this dataset exists in all backends and extract runtimes
            all_backends_have_data = True
            temp_runtimes = {}
            
            for backend_suffix in available_backends:
                if full_key not in backend_data[backend_suffix]:
                    all_backends_have_data = False
                    break
                    
                backend_datasets = backend_data[backend_suffix][full_key]['datasets']
                if dataset_key not in backend_datasets:
                    all_backends_have_data = False
                    break
                
                try:
                    # Calculate mean runtime in milliseconds
                    avg_ms = np.mean(backend_datasets[dataset_key]['runtimes']) / 1000.0
                    temp_runtimes[backend_suffix] = avg_ms
                except KeyError as e:
                    print(f"Error extracting runtimes for '{benchmark_name}' at {param_name}={param_value}: Key {e} not found. Skipping.")
                    all_backends_have_data = False
                    break
            
            if not all_backends_have_data:
                print(f"Warning: Dataset '{dataset_key}' for benchmark '{benchmark_name}' missing in some backends. Skipping.")
                continue
            
            # All backends have this data point
            sizes.append(param_value)
            for backend_suffix, runtime in temp_runtimes.items():
                backend_runtimes[backend_suffix].append(runtime)
            
        # 4. Process collected data
        if not sizes:
            continue
            
        # Sort data by parameter value
        sorted_indices = np.argsort(sizes)
        
        result_dict = {
            'sizes': np.array(sizes)[sorted_indices],
            'param_name': param_name,
        }
        
        # Store runtimes for each backend
        for backend_suffix in available_backends:
            result_dict[f'{backend_suffix}_runtimes'] = np.array(backend_runtimes[backend_suffix])[sorted_indices]
        
        # Calculate speedups relative to baseline
        if BASELINE_BACKEND in available_backends:
            baseline_times = result_dict[f'{BASELINE_BACKEND}_runtimes']
            for backend_suffix in available_backends:
                if backend_suffix != BASELINE_BACKEND:
                    backend_times = result_dict[f'{backend_suffix}_runtimes']
                    result_dict[f'{backend_suffix}_speedups'] = baseline_times / backend_times

        # 5. Store results
        all_results[benchmark_name] = result_dict

    return all_results

# --- Plotting Functions ---

def create_plot(benchmark_name: str,
                data: Dict[str, Any],
                available_backends: List[Backend],
                output_file: str,
                xbase: Optional[int] = None,
                ybase: Optional[int] = None):
    """Generates and saves the individual matplotlib figure."""
    
    sizes = data['sizes']
    param_name = data.get('param_name', 'parameter')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: Runtimes (Left Axis) ---
    plots = []
    for backend in available_backends:
        runtime_key = f'{backend.file_suffix}_runtimes'
        if runtime_key in data:
            p = ax1.plot(sizes, data[runtime_key], 
                        f'{backend.color}-{backend.marker}', 
                        label=f'{backend.name} runtime')
            plots.extend(p)
    
    ax1.set_xlabel(f'{param_name}')
    ax1.set_ylabel('Runtime (ms)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    
    # Set x-ticks to match the specific data sizes provided
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(int(s)) if s == int(s) else str(s) for s in sizes], rotation=45)
    
    if xbase is not None:
        ax1.set_xscale('log', base=xbase)

    if ybase is not None:
        ax1.set_yscale('log', base=ybase)

    # --- Plot 2: Speedup (Right Axis) ---
    ax2 = ax1.twinx()
    for backend in available_backends:
        speedup_key = f'{backend.file_suffix}_speedups'
        if speedup_key in data:
            p = ax2.plot(sizes, data[speedup_key], 
                        f'k-{backend.marker}', 
                        label=f'{backend.name} speedup',
                        alpha=0.7)
            plots.extend(p)
    
    ax2.set_ylabel('Speedup (x)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # --- Legend & Layout ---
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc='best')

    ax1.set_title(f'Benchmark: {benchmark_name}')
    fig.tight_layout()

    print(f"Saving plot for {benchmark_name} to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close(fig)

def create_combined_metric_plot(all_results: BenchmarkResults, 
                                backend: Backend,
                                output_file: str, 
                                xbase: Optional[int]):
    """
    Generates a single plot containing lines for ALL benchmarks 
    for a specific backend's runtimes.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_key = f'{backend.file_suffix}_runtimes'
    
    # Iterate through all benchmarks and plot them on the same axis
    for benchmark_name, data in all_results.items():
        if metric_key in data and data['sizes'].size > 0:
            param_name = data.get('param_name', 'parameter')
            ax.plot(data['sizes'], 
                    data[metric_key], 
                    marker='o', 
                    markersize=4, 
                    label=f"{benchmark_name} ({param_name})")

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title(f'Combined {backend.name} Runtimes - All Benchmarks')

    if xbase is not None:
        ax.set_xscale('log', base=xbase)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    fig.tight_layout()
    
    print(f"Saving combined plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close(fig)

# --- Main Execution ---

def main():
    args = parse_arguments()

    # 1. Load data for all available backends
    backend_data = {}
    available_backends = []
    
    for backend in AVAILABLE_BACKENDS:
        filepath = Path(f'{args.progname}-{backend.file_suffix}.json')
        data = load_json(filepath)
        if data is not None:
            backend_data[backend.file_suffix] = data
            available_backends.append(backend.file_suffix)
            print(f"Loaded data for backend: {backend.name}")
        else:
            print(f"Backend '{backend.name}' data file not found. Skipping this backend.")
    
    if not available_backends:
        print("Error: No backend data files found. Exiting.")
        sys.exit(1)
    
    # Check if baseline backend is available
    if BASELINE_BACKEND not in available_backends:
        print(f"Warning: Baseline backend '{BASELINE_BACKEND}' not found. Speedup calculations will be skipped.")

    # 2. Extract Results
    all_results = extract_all_benchmark_data(backend_data, available_backends)
    
    if not all_results:
        print("No complete benchmark data found to plot. Exiting.")
        sys.exit(0)

    # Get Backend objects for available backends
    available_backend_objs = [b for b in AVAILABLE_BACKENDS if b.file_suffix in available_backends]

    # 3. Plot Individual Results
    for benchmark_name, data in all_results.items():
        if data['sizes'].size > 0:
            create_plot(
                benchmark_name=benchmark_name,
                data=data,
                available_backends=available_backend_objs,
                output_file=f'{args.progname}-{benchmark_name}.png',
                xbase=args.xbase,
                ybase=args.ybase,
            )
        else:
            print(f"Warning: No valid data points to plot for benchmark '{benchmark_name}'.")

    # 4. Plot Combined Results for each backend
    print("\nGenerating combined plots...")
    for backend in available_backend_objs:
        create_combined_metric_plot(
            all_results, 
            backend, 
            f'{args.progname}-combined-{backend.file_suffix}.png',
            xbase=args.xbase
        )

    print("\nPlotting complete!")

if __name__ == '__main__':
    main()
