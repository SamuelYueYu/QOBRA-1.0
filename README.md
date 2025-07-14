<div align="center">
  <img src="assets/logo.png" alt="QOBRA Logo" width="250">
</div>

# QOBRA-PennyLane: Quantum Optimization Benchmark Library

A **PennyLane-based** recreation of the Quantum Optimization Benchmark Library (QOBLIB), featuring the "Intractable Decathlon" - ten optimization problem classes designed for benchmarking quantum optimization algorithms.

## 🌟 Overview

This project recreates the original QOBRA-1.0/QOBLIB (originally implemented with Qiskit) using **PennyLane**, providing a comprehensive benchmarking framework for quantum optimization algorithms. The library includes challenging optimization problems that become difficult for classical methods at relatively small problem sizes, making them ideal for testing near-term quantum devices.

## 📊 The Intractable Decathlon

The benchmark includes 10 optimization problem classes:

1. **Maximum Cut (Max-Cut)**
2. **Quadratic Unconstrained Binary Optimization (QUBO)**
3. **Maximum Independent Set**
4. **Traveling Salesman Problem (TSP)**
5. **Bin Packing**
6. **Portfolio Optimization**
7. **Maximum Satisfiability (Max-SAT)**
8. **Graph Coloring**
9. **Capacitated Vehicle Routing**
10. **Knapsack Problem**

## 🚀 Features

- **PennyLane Implementation**: Complete recreation using PennyLane for broader quantum framework compatibility
- **Quantum Algorithms**: QAOA, VQE, and other variational algorithms
- **Classical Baselines**: Reference implementations for performance comparison
- **Multiple Backends**: Support for various quantum simulators and hardware
- **Benchmarking Tools**: Standardized metrics and evaluation frameworks
- **Problem Generators**: Tools to create problem instances of varying difficulty

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/SamuelYueYu/QOBRA-PennyLane.git
cd QOBRA-PennyLane

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 📦 Requirements

- Python 3.8+
- PennyLane >= 0.33.0
- NumPy
- SciPy
- NetworkX
- Matplotlib
- Optional: JAX, TensorFlow, PyTorch (for different PennyLane backends)

## 🎯 Quick Start

```python
import pennylane as qml
from qobra_pennylane import MaxCutQAOA, generate_random_graph

# Generate a random Max-Cut problem
graph = generate_random_graph(nodes=6, edge_probability=0.6)

# Set up QAOA with PennyLane
dev = qml.device('default.qubit', wires=6)
qaoa = MaxCutQAOA(graph, device=dev, layers=2)

# Run optimization
result = qaoa.optimize(max_iterations=100)
print(f"Best cut value: {result.best_cost}")
print(f"Optimal parameters: {result.optimal_params}")
```

## 📁 Repository Structure

```
QOBRA-PennyLane/
├── qobra_pennylane/           # Main package
│   ├── algorithms/            # Quantum algorithms (QAOA, VQE, etc.)
│   ├── problems/              # Problem formulations
│   ├── benchmarks/            # Benchmarking utilities
│   ├── classical/             # Classical baseline solvers
│   └── utils/                 # Helper functions
├── examples/                  # Example notebooks and scripts
├── data/                      # Problem instances and datasets
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🧮 Algorithms Implemented

### Quantum Algorithms
- **QAOA** (Quantum Approximate Optimization Algorithm)
- **VQE** (Variational Quantum Eigensolver)
- **QIRO** (Quantum Iterative Routing Optimization)
- **Quantum Annealing** simulation

### Classical Algorithms
- Simulated Annealing
- Genetic Algorithm
- Tabu Search
- Gurobi/CPLEX integration

## 📈 Benchmarking

The library provides standardized benchmarking with:

- **Performance Metrics**: Approximation ratio, time-to-solution, success probability
- **Problem Scaling**: Automatic generation of problem instances of varying sizes
- **Comparison Tools**: Easy comparison between quantum and classical approaches
- **Visualization**: Built-in plotting for results analysis

## 🎓 Examples

Explore the `examples/` directory for comprehensive tutorials:

- `max_cut_tutorial.ipynb` - Introduction to Max-Cut with QAOA
- `portfolio_optimization.ipynb` - Financial portfolio optimization
- `tsp_comparison.ipynb` - Comparing quantum vs classical TSP solvers
- `benchmarking_guide.ipynb` - Complete benchmarking workflow

## 📊 Performance Tracking

All benchmark results can be submitted to track progress in quantum optimization:

```python
from qobra_pennylane.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_benchmark("max_cut", problem_sizes=[4, 6, 8, 10])
runner.save_results("my_benchmark_results.json")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Adding new problem formulations
- Implementing additional quantum algorithms
- Submitting benchmark results
- Improving documentation

## 📜 Citation

If you use QOBRA-PennyLane in your research, please cite:

```bibtex
@software{qobra_pennylane,
  title={QOBRA-PennyLane: Quantum Optimization Benchmark Library with PennyLane},
  author={Samuel Yu},
  year={2025},
  url={https://github.com/SamuelYueYu/QOBRA-PennyLane}
}
```

Original QOBLIB paper:
```bibtex
@article{qoblib2024,
  title={Quantum Optimization Benchmark Library -- The Intractable Decathlon},
  author={Koch, Thorsten and others},
  journal={arXiv preprint arXiv:2504.03832},
  year={2024}
}
```

## 🔗 Related Projects

- [Original QOBLIB](https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library) - The original Qiskit-based implementation
- [PennyLane](https://pennylane.ai/) - The quantum computing framework used in this project
- [Qiskit Optimization](https://qiskit.org/ecosystem/optimization/) - IBM's quantum optimization library

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original QOBLIB team for the foundational work
- PennyLane team for the excellent quantum computing framework
- Quantum Optimization Working Group for problem formulations
- Contributors and the quantum computing community

---

**Ready to benchmark quantum optimization? Let's push the boundaries of what's possible with near-term quantum devices! 🚀**
