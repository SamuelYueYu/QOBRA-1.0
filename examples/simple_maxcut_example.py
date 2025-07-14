#!/usr/bin/env python3
"""
Simple Max-Cut example using QOBRA-PennyLane

This script demonstrates how to solve a Max-Cut problem using QAOA
with the QOBRA-PennyLane library.
"""

import pennylane as qml
import numpy as np
import networkx as nx

# Import QOBRA-PennyLane (when installed)
try:
    from qobra_pennylane.algorithms.qaoa import QAOA
    from qobra_pennylane.problems.maxcut import MaxCut
    from qobra_pennylane.utils.problem_generators import generate_random_graph
except ImportError:
    print("QOBRA-PennyLane not installed. Please run 'pip install -e .' from the root directory.")
    exit(1)


def main():
    """Main example function."""
    print("🚀 QOBRA-PennyLane Max-Cut Example")
    print("=" * 50)
    
    # Step 1: Create a random graph
    print("\n📊 Step 1: Creating a random graph...")
    n_nodes = 6
    graph = generate_random_graph(n_nodes, edge_probability=0.6, seed=42)
    
    print(f"Generated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Step 2: Create Max-Cut problem
    print("\n🎯 Step 2: Setting up Max-Cut problem...")
    maxcut_problem = MaxCut(graph)
    
    # Get problem information
    problem_info = maxcut_problem.get_problem_info()
    print("Problem Information:")
    for key, value in problem_info.items():
        if key not in ['edge_weights', 'optimal_assignment']:  # Skip verbose output
            print(f"  {key}: {value}")
    
    # Convert to Hamiltonian
    cost_hamiltonian = maxcut_problem.to_hamiltonian()
    print(f"Cost Hamiltonian has {len(cost_hamiltonian.ops)} terms")
    
    # Step 3: Set up QAOA
    print("\n⚛️  Step 3: Setting up QAOA...")
    device = qml.device('default.qubit', wires=n_nodes, shots=1000)
    
    qaoa = QAOA(
        cost_hamiltonian=cost_hamiltonian,
        device=device,
        layers=2,  # Number of QAOA layers
        optimizer="COBYLA",
        seed=42
    )
    
    # Get circuit information
    circuit_info = qaoa.get_quantum_circuit_info()
    print("QAOA Circuit Information:")
    for key, value in circuit_info.items():
        print(f"  {key}: {value}")
    
    # Step 4: Run optimization
    print("\n🔧 Step 4: Running QAOA optimization...")
    
    def progress_callback(params):
        if len(qaoa.cost_history) % 10 == 0:
            print(f"  Iteration {len(qaoa.cost_history)}: Cost = {qaoa.cost_history[-1]:.4f}")
    
    result = qaoa.optimize(
        max_iterations=50,
        callback=progress_callback
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"Success: {result.success}")
    print(f"Best cost: {result.best_cost:.4f}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    print(f"Function evaluations: {result.function_evaluations}")
    
    # Step 5: Analyze results
    print("\n📈 Step 5: Analyzing results...")
    
    # Sample bitstrings from the optimized state
    bitstrings, probabilities = qaoa.sample_bitstrings(result.optimal_params, n_samples=1000)
    
    # Convert to cut solutions
    cut_values = []
    solutions = []
    for bitstring in bitstrings:
        solution = maxcut_problem.bitstring_to_cut(bitstring)
        cut_values.append(solution.cut_value)
        solutions.append(solution)
    
    # Find best solution
    best_idx = np.argmax(cut_values)
    best_solution = solutions[best_idx]
    
    print(f"Best solution found:")
    print(f"  Cut value: {best_solution.cut_value}")
    print(f"  Cut assignment: {best_solution.cut_assignment}")
    print(f"  Number of cut edges: {len(best_solution.cut_edges)}")
    if best_solution.approximation_ratio is not None:
        print(f"  Approximation ratio: {best_solution.approximation_ratio:.4f}")
    
    # Step 6: Compare with classical methods
    print("\n🏆 Step 6: Comparing with classical methods...")
    
    # Get classical baselines
    random_solution = maxcut_problem.get_random_solution(seed=42)
    greedy_solution = maxcut_problem.get_greedy_solution()
    
    print("Solution Comparison:")
    print(f"  Random solution:  {random_solution.cut_value}")
    print(f"  Greedy solution:  {greedy_solution.cut_value}")
    print(f"  QAOA solution:    {best_solution.cut_value}")
    
    if maxcut_problem.instance.optimal_value is not None:
        print(f"  Optimal solution: {maxcut_problem.instance.optimal_value}")
        print(f"\nApproximation ratios:")
        print(f"  Random:  {random_solution.cut_value / maxcut_problem.instance.optimal_value:.4f}")
        print(f"  Greedy:  {greedy_solution.cut_value / maxcut_problem.instance.optimal_value:.4f}")
        print(f"  QAOA:    {best_solution.cut_value / maxcut_problem.instance.optimal_value:.4f}")
    
    # Step 7: Show final parameters
    print(f"\n🎛️  Step 7: Final QAOA parameters:")
    print(f"Gamma (cost): {result.optimal_params[:qaoa.layers]}")
    print(f"Beta (mixer): {result.optimal_params[qaoa.layers:]}")
    
    print("\n🎉 Example completed successfully!")
    print("\nTo visualize results and see more detailed analysis,")
    print("check out the Jupyter notebooks in the examples/ directory!")


if __name__ == "__main__":
    main()