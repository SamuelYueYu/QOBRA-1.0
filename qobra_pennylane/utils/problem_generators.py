"""
Problem generators for creating benchmark instances.

This module provides functions to generate various types of graphs, QUBO problems,
and other optimization problem instances for benchmarking quantum algorithms.
"""

import numpy as np
import networkx as nx
from typing import Optional, Dict, Tuple, List, Union
import random


def generate_random_graph(
    n_nodes: int,
    edge_probability: float = 0.5,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
    connected: bool = True,
) -> nx.Graph:
    """
    Generate a random graph using the Erdős-Rényi model.
    
    Args:
        n_nodes: Number of nodes in the graph
        edge_probability: Probability of edge creation between any two nodes
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        connected: Whether to ensure the graph is connected
        
    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate basic random graph
    G = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=seed)
    
    # Ensure connectivity if requested
    if connected and not nx.is_connected(G):
        # Add edges to make it connected
        components = list(nx.connected_components(G))
        while len(components) > 1:
            # Connect first two components
            comp1 = list(components[0])
            comp2 = list(components[1])
            
            node1 = random.choice(comp1)
            node2 = random.choice(comp2)
            G.add_edge(node1, node2)
            
            components = list(nx.connected_components(G))
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        # All weights the same
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_regular_graph(
    n_nodes: int,
    degree: int,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a random regular graph where all nodes have the same degree.
    
    Args:
        n_nodes: Number of nodes
        degree: Degree of each node
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX regular graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Check feasibility
    if n_nodes * degree % 2 != 0:
        raise ValueError("n_nodes * degree must be even for a regular graph")
    
    if degree >= n_nodes:
        raise ValueError("Degree must be less than number of nodes")
    
    # Generate regular graph
    G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_erdos_renyi_graph(
    n_nodes: int,
    n_edges: Optional[int] = None,
    edge_probability: Optional[float] = None,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate an Erdős-Rényi random graph.
    
    Args:
        n_nodes: Number of nodes
        n_edges: Number of edges (alternative to edge_probability)
        edge_probability: Probability of edge creation
        weight_range: Range for random edge weights (min, max)  
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if n_edges is not None:
        G = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    elif edge_probability is not None:
        G = nx.gnp_random_graph(n_nodes, edge_probability, seed=seed)
    else:
        raise ValueError("Must specify either n_edges or edge_probability")
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_complete_graph(
    n_nodes: int,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a complete graph where every pair of nodes is connected.
    
    Args:
        n_nodes: Number of nodes
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX complete graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    G = nx.complete_graph(n_nodes)
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_cycle_graph(
    n_nodes: int,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a cycle graph.
    
    Args:
        n_nodes: Number of nodes
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX cycle graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    G = nx.cycle_graph(n_nodes)
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_path_graph(
    n_nodes: int,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a path graph.
    
    Args:
        n_nodes: Number of nodes
        weight_range: Range for random edge weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX path graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    G = nx.path_graph(n_nodes)
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_grid_graph(
    m: int,
    n: int,
    weight_range: Tuple[float, float] = (1.0, 1.0),
    periodic: bool = False,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a 2D grid graph.
    
    Args:
        m: Number of rows
        n: Number of columns
        weight_range: Range for random edge weights (min, max)
        periodic: Whether to use periodic boundary conditions
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX grid graph
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if periodic:
        G = nx.grid_2d_graph(m, n, periodic=True)
    else:
        G = nx.grid_2d_graph(m, n)
    
    # Convert to integer node labels
    G = nx.convert_node_labels_to_integers(G)
    
    # Add random weights
    if weight_range[0] != weight_range[1]:
        for u, v in G.edges():
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G[u][v]['weight'] = weight
    else:
        for u, v in G.edges():
            G[u][v]['weight'] = weight_range[0]
    
    return G


def generate_random_qubo(
    n_variables: int,
    density: float = 0.5,
    coefficient_range: Tuple[float, float] = (-1.0, 1.0),
    diagonal_range: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Generate a random QUBO problem instance.
    
    Args:
        n_variables: Number of binary variables
        density: Density of non-zero coefficients
        coefficient_range: Range for quadratic coefficients
        diagonal_range: Range for linear coefficients (diagonal terms)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (Q matrix, constant offset)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if diagonal_range is None:
        diagonal_range = coefficient_range
    
    Q = np.zeros((n_variables, n_variables))
    
    # Add quadratic terms (off-diagonal)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if np.random.random() < density:
                coeff = np.random.uniform(coefficient_range[0], coefficient_range[1])
                Q[i, j] = coeff
                Q[j, i] = coeff  # Symmetric
    
    # Add linear terms (diagonal)
    for i in range(n_variables):
        if np.random.random() < density:
            coeff = np.random.uniform(diagonal_range[0], diagonal_range[1])
            Q[i, i] = coeff
    
    # Random constant offset
    offset = np.random.uniform(-1.0, 1.0)
    
    return Q, offset


def generate_maxcut_instances(
    sizes: List[int],
    graph_types: List[str] = ["random", "regular", "complete"],
    instances_per_size: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, List[nx.Graph]]:
    """
    Generate a collection of Max-Cut problem instances.
    
    Args:
        sizes: List of graph sizes to generate
        graph_types: Types of graphs to generate
        instances_per_size: Number of instances per size/type combination
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping instance names to graphs
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    instances = {}
    
    for size in sizes:
        for graph_type in graph_types:
            for instance_id in range(instances_per_size):
                key = f"{graph_type}_{size}_{instance_id}"
                
                if graph_type == "random":
                    # Random graph with edge probability 0.5
                    graph = generate_random_graph(
                        size, 
                        edge_probability=0.5,
                        seed=seed + hash(key) if seed else None
                    )
                
                elif graph_type == "regular":
                    # 3-regular graph (if possible)
                    degree = min(3, size - 1)
                    if size * degree % 2 == 0:  # Check feasibility
                        graph = generate_regular_graph(
                            size, 
                            degree,
                            seed=seed + hash(key) if seed else None
                        )
                    else:
                        # Fallback to random graph
                        graph = generate_random_graph(
                            size,
                            edge_probability=0.5,
                            seed=seed + hash(key) if seed else None
                        )
                
                elif graph_type == "complete":
                    graph = generate_complete_graph(
                        size,
                        seed=seed + hash(key) if seed else None
                    )
                
                elif graph_type == "cycle":
                    graph = generate_cycle_graph(
                        size,
                        seed=seed + hash(key) if seed else None
                    )
                
                elif graph_type == "path":
                    graph = generate_path_graph(
                        size,
                        seed=seed + hash(key) if seed else None
                    )
                
                elif graph_type == "grid":
                    # Create square-ish grid
                    m = int(np.sqrt(size))
                    n = size // m
                    if m * n < size:
                        n += 1
                    graph = generate_grid_graph(
                        m, n,
                        seed=seed + hash(key) if seed else None
                    )
                    # Remove extra nodes if needed
                    if graph.number_of_nodes() > size:
                        nodes_to_remove = list(range(size, graph.number_of_nodes()))
                        graph.remove_nodes_from(nodes_to_remove)
                        graph = nx.convert_node_labels_to_integers(graph)
                
                else:
                    raise ValueError(f"Unknown graph type: {graph_type}")
                
                instances[key] = graph
    
    return instances


def generate_benchmark_suite(
    problem_types: List[str] = ["maxcut"],
    sizes: List[int] = [4, 6, 8, 10, 12],
    instances_per_type: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Generate a comprehensive benchmark suite.
    
    Args:
        problem_types: Types of problems to include
        sizes: Problem sizes to generate
        instances_per_type: Number of instances per problem type/size
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all benchmark instances
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    benchmark_suite = {}
    
    for problem_type in problem_types:
        if problem_type == "maxcut":
            # Generate Max-Cut instances
            graph_types = ["random", "regular", "complete", "cycle"]
            instances = generate_maxcut_instances(
                sizes, 
                graph_types,
                instances_per_type // len(graph_types),
                seed
            )
            benchmark_suite["maxcut"] = instances
        
        elif problem_type == "qubo":
            # Generate random QUBO instances
            instances = {}
            for size in sizes:
                for instance_id in range(instances_per_type):
                    key = f"qubo_{size}_{instance_id}"
                    Q, offset = generate_random_qubo(
                        size,
                        density=0.5,
                        seed=seed + hash(key) if seed else None
                    )
                    instances[key] = {"Q": Q, "offset": offset}
            benchmark_suite["qubo"] = instances
        
        # Add more problem types as needed
    
    return benchmark_suite