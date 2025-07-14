"""
Maximum Cut (Max-Cut) problem implementation.

The Max-Cut problem seeks to partition the vertices of a graph into two sets
such that the number of edges between the sets is maximized.
"""

import pennylane as qml
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass 
class MaxCutInstance:
    """Container for Max-Cut problem instance data."""
    graph: nx.Graph
    n_vertices: int
    n_edges: int
    edge_weights: Dict[Tuple[int, int], float]
    adjacency_matrix: np.ndarray
    optimal_value: Optional[float] = None
    optimal_cut: Optional[List[int]] = None


@dataclass
class MaxCutSolution:
    """Container for Max-Cut solution."""
    cut_assignment: List[int]  # 0 or 1 for each vertex
    cut_value: float
    cut_edges: List[Tuple[int, int]]
    approximation_ratio: Optional[float] = None


class MaxCut:
    """
    Maximum Cut problem implementation for quantum optimization.
    
    The Max-Cut problem is formulated as finding a binary assignment x ∈ {0,1}^n
    that maximizes:
        C(x) = Σ_{(i,j) ∈ E} w_{ij} * x_i * (1 - x_j) + (1 - x_i) * x_j
    
    This is equivalent to maximizing:
        C(x) = Σ_{(i,j) ∈ E} w_{ij} * (1 - x_i * x_j)
    
    For quantum algorithms, we map x_i → (1 - Z_i)/2, giving:
        H = Σ_{(i,j) ∈ E} w_{ij}/2 * (I - Z_i ⊗ Z_j)
    
    Since we want to maximize, we minimize -H.
    """
    
    def __init__(
        self,
        graph: Union[nx.Graph, np.ndarray, List[Tuple[int, int]]],
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        """
        Initialize Max-Cut problem.
        
        Args:
            graph: Graph as NetworkX graph, adjacency matrix, or edge list
            edge_weights: Optional edge weights (default: all weights = 1)
        """
        self.graph = self._process_graph_input(graph, edge_weights)
        self.n_vertices = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        
        # Extract edge weights
        self.edge_weights = {}
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            self.edge_weights[(min(u, v), max(u, v))] = weight
        
        # Create adjacency matrix
        self.adjacency_matrix = nx.adjacency_matrix(self.graph, weight='weight').toarray()
        
        # Problem instance
        self.instance = MaxCutInstance(
            graph=self.graph,
            n_vertices=self.n_vertices,
            n_edges=self.n_edges,
            edge_weights=self.edge_weights,
            adjacency_matrix=self.adjacency_matrix,
        )
        
        # Compute optimal solution if small enough
        if self.n_vertices <= 20:  # Brute force limit
            self._compute_optimal_solution()
    
    def _process_graph_input(
        self,
        graph: Union[nx.Graph, np.ndarray, List[Tuple[int, int]]],
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> nx.Graph:
        """Process different graph input formats."""
        
        if isinstance(graph, nx.Graph):
            return graph.copy()
        
        elif isinstance(graph, np.ndarray):
            # Adjacency matrix
            G = nx.from_numpy_array(graph)
            return G
        
        elif isinstance(graph, list):
            # Edge list
            G = nx.Graph()
            
            # Add edges
            for edge in graph:
                if len(edge) == 2:
                    u, v = edge
                    weight = 1.0
                    if edge_weights and (min(u, v), max(u, v)) in edge_weights:
                        weight = edge_weights[(min(u, v), max(u, v))]
                    G.add_edge(u, v, weight=weight)
                elif len(edge) == 3:
                    u, v, weight = edge
                    G.add_edge(u, v, weight=weight)
                else:
                    raise ValueError(f"Invalid edge format: {edge}")
            
            return G
        
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    def _compute_optimal_solution(self):
        """Compute optimal Max-Cut solution via brute force (for small graphs)."""
        if self.n_vertices > 20:
            return  # Too large for brute force
        
        best_cut_value = -1
        best_assignment = None
        
        # Try all possible cuts (2^n possibilities)
        for i in range(2**self.n_vertices):
            assignment = [(i >> j) & 1 for j in range(self.n_vertices)]
            cut_value = self.evaluate_cut(assignment)
            
            if cut_value > best_cut_value:
                best_cut_value = cut_value
                best_assignment = assignment
        
        self.instance.optimal_value = best_cut_value
        self.instance.optimal_cut = best_assignment
    
    def to_hamiltonian(self) -> qml.Hamiltonian:
        """
        Convert Max-Cut problem to Hamiltonian formulation.
        
        Returns:
            PennyLane Hamiltonian representing the Max-Cut cost function
        """
        coeffs = []
        ops = []
        
        # Add constant term and ZZ interactions
        constant_term = 0
        
        for (u, v), weight in self.edge_weights.items():
            # For maximizing, we want to minimize the negative
            # H = -1/2 * Σ w_ij (I - Z_i Z_j)
            #   = -1/2 * Σ w_ij * I + 1/2 * Σ w_ij * Z_i Z_j
            
            constant_term -= weight / 2
            coeffs.append(weight / 2)
            ops.append(qml.PauliZ(u) @ qml.PauliZ(v))
        
        # Add constant term as identity
        if constant_term != 0:
            coeffs.append(constant_term)
            ops.append(qml.Identity(0))
        
        return qml.Hamiltonian(coeffs, ops)
    
    def to_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Convert Max-Cut to QUBO (Quadratic Unconstrained Binary Optimization) format.
        
        Returns:
            Tuple of (Q matrix, constant offset)
        """
        Q = np.zeros((self.n_vertices, self.n_vertices))
        offset = 0
        
        for (u, v), weight in self.edge_weights.items():
            # Max-Cut QUBO: maximize Σ w_ij (x_i + x_j - 2*x_i*x_j)
            # This becomes: maximize Σ w_ij x_i + Σ w_ij x_j - 2*Σ w_ij x_i x_j
            Q[u, u] += weight  # Linear term for u
            Q[v, v] += weight  # Linear term for v  
            Q[u, v] -= 2 * weight  # Quadratic interaction
            Q[v, u] -= 2 * weight  # Symmetric
        
        return Q, offset
    
    def evaluate_cut(self, assignment: List[int]) -> float:
        """
        Evaluate the cut value for a given assignment.
        
        Args:
            assignment: Binary assignment (0 or 1) for each vertex
            
        Returns:
            Cut value (sum of weights of edges crossing the cut)
        """
        cut_value = 0.0
        
        for (u, v), weight in self.edge_weights.items():
            if assignment[u] != assignment[v]:  # Edge crosses the cut
                cut_value += weight
        
        return cut_value
    
    def bitstring_to_cut(self, bitstring: str) -> MaxCutSolution:
        """
        Convert a bitstring to a Max-Cut solution.
        
        Args:
            bitstring: Binary string representing vertex assignments
            
        Returns:
            MaxCutSolution object
        """
        assignment = [int(bit) for bit in bitstring]
        cut_value = self.evaluate_cut(assignment)
        
        # Find edges that cross the cut
        cut_edges = []
        for (u, v), weight in self.edge_weights.items():
            if assignment[u] != assignment[v]:
                cut_edges.append((u, v))
        
        # Calculate approximation ratio if optimal is known
        approx_ratio = None
        if self.instance.optimal_value is not None and self.instance.optimal_value > 0:
            approx_ratio = cut_value / self.instance.optimal_value
        
        return MaxCutSolution(
            cut_assignment=assignment,
            cut_value=cut_value,
            cut_edges=cut_edges,
            approximation_ratio=approx_ratio,
        )
    
    def get_random_solution(self, seed: Optional[int] = None) -> MaxCutSolution:
        """Generate a random Max-Cut solution."""
        if seed is not None:
            np.random.seed(seed)
        
        assignment = np.random.randint(0, 2, self.n_vertices).tolist()
        return self.bitstring_to_cut(''.join(map(str, assignment)))
    
    def get_greedy_solution(self) -> MaxCutSolution:
        """Get a greedy Max-Cut solution."""
        assignment = [0] * self.n_vertices
        
        # Greedy: for each vertex, assign it to the side that maximizes cut value
        for v in range(self.n_vertices):
            # Try assigning v to side 0
            assignment[v] = 0
            cut_value_0 = self.evaluate_cut(assignment)
            
            # Try assigning v to side 1  
            assignment[v] = 1
            cut_value_1 = self.evaluate_cut(assignment)
            
            # Keep the better assignment
            if cut_value_0 >= cut_value_1:
                assignment[v] = 0
            else:
                assignment[v] = 1
        
        return self.bitstring_to_cut(''.join(map(str, assignment)))
    
    def visualize_cut(self, solution: MaxCutSolution, **kwargs):
        """
        Visualize the Max-Cut solution.
        
        Args:
            solution: MaxCutSolution to visualize
            **kwargs: Additional arguments for matplotlib/networkx plotting
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create color map for vertices
            colors = ['red' if x == 0 else 'blue' for x in solution.cut_assignment]
            
            # Draw graph
            pos = nx.spring_layout(self.graph)
            
            # Draw vertices
            nx.draw_networkx_nodes(self.graph, pos, node_color=colors, **kwargs)
            
            # Draw edges (cut edges in bold)
            regular_edges = []
            cut_edges = []
            
            for u, v in self.graph.edges():
                if (u, v) in solution.cut_edges or (v, u) in solution.cut_edges:
                    cut_edges.append((u, v))
                else:
                    regular_edges.append((u, v))
            
            # Draw regular edges
            nx.draw_networkx_edges(self.graph, pos, edgelist=regular_edges, 
                                 alpha=0.3, **kwargs)
            
            # Draw cut edges
            nx.draw_networkx_edges(self.graph, pos, edgelist=cut_edges,
                                 width=3, edge_color='black', **kwargs)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, **kwargs)
            
            plt.title(f"Max-Cut Solution (Cut Value: {solution.cut_value})")
            plt.axis('off')
            
            if 'save_path' in kwargs:
                plt.savefig(kwargs['save_path'])
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for visualization")
    
    def get_problem_info(self) -> Dict:
        """Get information about the Max-Cut problem instance."""
        info = {
            "problem_type": "MaxCut",
            "n_vertices": self.n_vertices,
            "n_edges": self.n_edges,
            "edge_weights": self.edge_weights,
            "total_weight": sum(self.edge_weights.values()),
            "is_weighted": len(set(self.edge_weights.values())) > 1,
            "is_complete": self.n_edges == self.n_vertices * (self.n_vertices - 1) // 2,
        }
        
        if self.instance.optimal_value is not None:
            info["optimal_cut_value"] = self.instance.optimal_value
            info["optimal_assignment"] = self.instance.optimal_cut
        
        return info