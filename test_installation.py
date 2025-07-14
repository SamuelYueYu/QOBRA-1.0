#!/usr/bin/env python3
"""
Test script to verify QOBRA-PennyLane installation and basic functionality.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pennylane as qml
        print("  ✅ PennyLane imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import PennyLane: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import NumPy: {e}")
        return False
    
    try:
        import networkx as nx
        print("  ✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import NetworkX: {e}")
        return False
    
    try:
        # Test main package import
        import qobra_pennylane
        print("  ✅ QOBRA-PennyLane package imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import QOBRA-PennyLane: {e}")
        print("  💡 Make sure to run 'pip install -e .' from the repository root")
        return False
    
    try:
        # Test algorithm imports
        from qobra_pennylane.algorithms.qaoa import QAOA
        print("  ✅ QAOA algorithm imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import QAOA: {e}")
        return False
    
    try:
        # Test problem imports
        from qobra_pennylane.problems.maxcut import MaxCut
        print("  ✅ MaxCut problem imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import MaxCut: {e}")
        return False
    
    try:
        # Test utilities imports
        from qobra_pennylane.utils.problem_generators import generate_random_graph
        from qobra_pennylane.utils.device_utils import create_device
        print("  ✅ Utility functions imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import utilities: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with a small example."""
    print("\n🔬 Testing basic functionality...")
    
    try:
        from qobra_pennylane.utils.problem_generators import generate_random_graph
        from qobra_pennylane.problems.maxcut import MaxCut
        from qobra_pennylane.algorithms.qaoa import QAOA
        from qobra_pennylane.utils.device_utils import create_device
        import pennylane as qml
        
        # Create a small graph
        graph = generate_random_graph(4, edge_probability=0.6, seed=42)
        print(f"  ✅ Generated random graph with {graph.number_of_nodes()} nodes")
        
        # Create Max-Cut problem
        maxcut = MaxCut(graph)
        print(f"  ✅ Created Max-Cut problem")
        
        # Convert to Hamiltonian
        hamiltonian = maxcut.to_hamiltonian()
        print(f"  ✅ Converted to Hamiltonian with {len(hamiltonian.ops)} terms")
        
        # Create device
        device = create_device("default.qubit", wires=4, shots=1000)
        print(f"  ✅ Created quantum device: {device}")
        
        # Create QAOA instance
        qaoa = QAOA(
            cost_hamiltonian=hamiltonian,
            device=device,
            layers=1,
            seed=42
        )
        print(f"  ✅ Created QAOA instance")
        
        # Get circuit info
        info = qaoa.get_quantum_circuit_info()
        print(f"  ✅ Circuit info: {info['n_qubits']} qubits, {info['n_parameters']} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_optimization():
    """Test a minimal optimization run."""
    print("\n⚡ Testing QAOA optimization...")
    
    try:
        from qobra_pennylane.utils.problem_generators import generate_random_graph
        from qobra_pennylane.problems.maxcut import MaxCut
        from qobra_pennylane.algorithms.qaoa import QAOA
        import pennylane as qml
        import numpy as np
        
        # Create a tiny problem
        graph = generate_random_graph(3, edge_probability=1.0, seed=42)
        maxcut = MaxCut(graph)
        hamiltonian = maxcut.to_hamiltonian()
        
        # Create device and QAOA
        device = qml.device("default.qubit", wires=3, shots=100)
        qaoa = QAOA(
            cost_hamiltonian=hamiltonian,
            device=device,
            layers=1,
            seed=42
        )
        
        # Run a very short optimization
        result = qaoa.optimize(max_iterations=5)
        
        print(f"  ✅ Optimization completed: success={result.success}")
        print(f"  ✅ Best cost: {result.best_cost:.4f}")
        print(f"  ✅ Function evaluations: {result.function_evaluations}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization test failed: {e}")
        traceback.print_exc()
        return False


def test_problem_generators():
    """Test problem generation utilities."""
    print("\n🎲 Testing problem generators...")
    
    try:
        from qobra_pennylane.utils.problem_generators import (
            generate_random_graph,
            generate_regular_graph,
            generate_complete_graph,
            generate_random_qubo
        )
        
        # Test graph generators
        random_graph = generate_random_graph(5, edge_probability=0.5, seed=42)
        print(f"  ✅ Random graph: {random_graph.number_of_nodes()} nodes, {random_graph.number_of_edges()} edges")
        
        regular_graph = generate_regular_graph(6, degree=2, seed=42)
        print(f"  ✅ Regular graph: {regular_graph.number_of_nodes()} nodes, degree 2")
        
        complete_graph = generate_complete_graph(4, seed=42)
        print(f"  ✅ Complete graph: {complete_graph.number_of_nodes()} nodes, {complete_graph.number_of_edges()} edges")
        
        # Test QUBO generator
        Q, offset = generate_random_qubo(4, density=0.5, seed=42)
        print(f"  ✅ Random QUBO: {Q.shape} matrix, offset={offset:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Problem generator test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 QOBRA-PennyLane Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Optimization Test", test_optimization),
        ("Problem Generators Test", test_problem_generators),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! QOBRA-PennyLane is working correctly.")
        print("\n🚀 Ready to benchmark quantum optimization algorithms!")
        print("\nNext steps:")
        print("  1. Check out examples/simple_maxcut_example.py")
        print("  2. Explore the Jupyter notebooks in examples/")
        print("  3. Read the documentation in docs/")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("  1. Make sure you've installed all dependencies: pip install -r requirements.txt")
        print("  2. Install the package in development mode: pip install -e .")
        print("  3. Check that PennyLane is properly installed: pip install pennylane")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)