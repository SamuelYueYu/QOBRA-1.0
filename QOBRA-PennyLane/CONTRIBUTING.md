# Contributing to QOBRA-PennyLane

Welcome to QOBRA-PennyLane! We're excited to have you contribute to the quantum optimization benchmarking community.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PennyLane >= 0.33.0
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/SamuelYueYu/QOBRA-PennyLane.git
   cd QOBRA-PennyLane
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv qobra_env
   source qobra_env/bin/activate  # On Windows: qobra_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run tests**
   ```bash
   python test_installation.py
   ```

## 🎯 How to Contribute

### 1. Adding New Problem Formulations

To add a new optimization problem to the "Intractable Decathlon":

1. Create a new file in `qobra_pennylane/problems/`
2. Implement your problem class following the pattern in `maxcut.py`
3. Add conversion methods to Hamiltonian and QUBO formats
4. Include classical baseline methods
5. Add your problem to `qobra_pennylane/problems/__init__.py`

**Example structure:**
```python
class NewProblem:
    def __init__(self, problem_data):
        # Initialize problem
        pass
    
    def to_hamiltonian(self) -> qml.Hamiltonian:
        # Convert to PennyLane Hamiltonian
        pass
    
    def evaluate_solution(self, solution):
        # Evaluate solution quality
        pass
    
    def get_classical_solution(self):
        # Classical baseline
        pass
```

### 2. Implementing New Quantum Algorithms

To add a new quantum optimization algorithm:

1. Create a new file in `qobra_pennylane/algorithms/`
2. Follow the pattern established in `qaoa.py`
3. Include optimization result classes
4. Add comprehensive documentation
5. Update `qobra_pennylane/algorithms/__init__.py`

### 3. Adding Benchmarking Tools

Enhance the benchmarking framework by:

1. Adding new metrics in `qobra_pennylane/utils/metrics.py`
2. Creating visualization tools in `qobra_pennylane/utils/visualization.py`
3. Implementing problem generators in `qobra_pennylane/utils/problem_generators.py`

### 4. Documentation and Examples

- Add Jupyter notebooks to `examples/`
- Update README.md with new features
- Include docstrings following NumPy style
- Add type hints where applicable

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Include comprehensive docstrings
- Add type hints for function parameters and returns

### Testing

- Test your code thoroughly before submitting
- Include unit tests for new functionality
- Verify examples work correctly
- Run the installation test: `python test_installation.py`

### Documentation

- Document all public functions and classes
- Include usage examples in docstrings
- Update README.md if adding major features
- Create tutorial notebooks for complex features

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   python test_installation.py
   python examples/simple_maxcut_example.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include examples of new functionality

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Paper acknowledgments (if applicable)

## 📝 Submitting Benchmark Results

We encourage submission of benchmark results! To contribute:

1. **Run benchmarks** using the provided framework
2. **Document your setup**: hardware, software versions, parameters
3. **Submit results** via pull request to `data/benchmark_results/`
4. **Follow the naming convention**: `algorithm_problem_size_date.json`

## 🚨 Reporting Issues

When reporting bugs or issues:

1. **Check existing issues** first
2. **Provide minimal reproduction** example
3. **Include system information**: OS, Python version, PennyLane version
4. **Describe expected vs actual behavior**

## 💡 Feature Requests

For feature requests:

1. **Check if it already exists**
2. **Explain the use case** and motivation
3. **Provide implementation suggestions** if possible
4. **Consider contributing** the feature yourself!

## 📚 Resources

- [PennyLane Documentation](https://docs.pennylane.ai/)
- [Original QOBLIB Paper](https://arxiv.org/abs/2504.03832)
- [Quantum Optimization Working Group](https://quantumconsortium.org/quantum-working-groups/)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a welcoming environment

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: samuel.yu@yale.edu for direct contact

Thank you for contributing to QOBRA-PennyLane and advancing quantum optimization research! 🚀