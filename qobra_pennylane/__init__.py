"""
QOBRA-PennyLane: Quantum Optimization Benchmark Library with PennyLane

A comprehensive benchmarking framework for quantum optimization algorithms,
recreating the original QOBLIB using PennyLane instead of Qiskit.

Copyright 2025 Samuel Yu
Licensed under the Apache License, Version 2.0
"""

__version__ = "1.0.0"
__author__ = "Samuel Yu"
__email__ = "samuel.yu@yale.edu"

# Import main algorithm classes
from .algorithms import (
    QAOA,
    VQE,
    QIRO,
    QuantumAnnealing,
)

# Import problem formulations
from .problems import (
    MaxCut,
    QUBO,
    TSP,
    MaxIndependentSet,
    BinPacking,
    PortfolioOptimization,
    MaxSAT,
    GraphColoring,
    VehicleRouting,
    Knapsack,
)

# Import benchmarking utilities
from .benchmarks import (
    BenchmarkRunner,
    MetricsCalculator,
    ResultsAnalyzer,
)

# Import classical baselines
from .classical import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    TabuSearch,
)

# Import utility functions
from .utils import (
    generate_random_graph,
    create_device,
    load_problem_instance,
    save_results,
    visualize_results,
)

# Define what gets imported with "from qobra_pennylane import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Quantum algorithms
    "QAOA",
    "VQE", 
    "QIRO",
    "QuantumAnnealing",
    
    # Problem classes
    "MaxCut",
    "QUBO",
    "TSP",
    "MaxIndependentSet",
    "BinPacking",
    "PortfolioOptimization",
    "MaxSAT",
    "GraphColoring",
    "VehicleRouting",
    "Knapsack",
    
    # Benchmarking
    "BenchmarkRunner",
    "MetricsCalculator",
    "ResultsAnalyzer",
    
    # Classical algorithms
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    
    # Utilities
    "generate_random_graph",
    "create_device",
    "load_problem_instance",
    "save_results",
    "visualize_results",
]

# Package metadata
DESCRIPTION = "Quantum Optimization Benchmark Library with PennyLane"
LONG_DESCRIPTION = """
QOBRA-PennyLane is a comprehensive benchmarking framework for quantum optimization 
algorithms, featuring the "Intractable Decathlon" - ten optimization problem classes 
designed for benchmarking quantum optimization algorithms. This library recreates 
the original QOBLIB using PennyLane instead of Qiskit, providing broader 
compatibility with different quantum frameworks.

Key Features:
- Implementation of QAOA, VQE, QIRO, and other quantum algorithms
- Ten challenging optimization problem classes
- Classical baseline algorithms for comparison
- Comprehensive benchmarking and metrics tools
- Support for multiple quantum devices and simulators
- Easy-to-use API for researchers and practitioners
"""

URL = "https://github.com/SamuelYueYu/QOBRA-PennyLane"
LICENSE = "Apache License 2.0"

# Package configuration
DEFAULT_DEVICE = "default.qubit"
DEFAULT_SHOTS = 1000
DEFAULT_SEED = 42

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())