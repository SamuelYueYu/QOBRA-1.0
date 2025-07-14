"""
Utility functions for the QOBRA-PennyLane library.

This module provides helper functions for device creation, problem generation,
result saving/loading, and visualization.
"""

from .device_utils import create_device
from .problem_generators import (
    generate_random_graph,
    generate_regular_graph,
    generate_erdos_renyi_graph,
    generate_complete_graph,
    generate_random_qubo,
)
from .io_utils import (
    load_problem_instance,
    save_results,
    load_results,
    export_to_json,
    import_from_json,
)
from .visualization import (
    visualize_results,
    plot_optimization_history,
    plot_graph,
    plot_hamiltonian_spectrum,
)
from .metrics import (
    calculate_approximation_ratio,
    calculate_success_probability,
    analyze_bitstring_distribution,
)

__all__ = [
    # Device utilities
    "create_device",
    
    # Problem generators
    "generate_random_graph",
    "generate_regular_graph", 
    "generate_erdos_renyi_graph",
    "generate_complete_graph",
    "generate_random_qubo",
    
    # I/O utilities
    "load_problem_instance",
    "save_results",
    "load_results",
    "export_to_json",
    "import_from_json",
    
    # Visualization
    "visualize_results",
    "plot_optimization_history",
    "plot_graph",
    "plot_hamiltonian_spectrum",
    
    # Metrics
    "calculate_approximation_ratio",
    "calculate_success_probability",
    "analyze_bitstring_distribution",
]