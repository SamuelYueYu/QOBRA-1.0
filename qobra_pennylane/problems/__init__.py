"""
Optimization problem formulations for the QOBRA-PennyLane benchmark library.

This module contains implementations of the "Intractable Decathlon" - ten 
optimization problem classes designed for benchmarking quantum optimization
algorithms.
"""

from .maxcut import MaxCut
from .qubo import QUBO
from .tsp import TSP
from .max_independent_set import MaxIndependentSet
from .bin_packing import BinPacking
from .portfolio_optimization import PortfolioOptimization
from .max_sat import MaxSAT
from .graph_coloring import GraphColoring
from .vehicle_routing import VehicleRouting
from .knapsack import Knapsack

__all__ = [
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
]