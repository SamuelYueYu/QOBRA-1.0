"""
Quantum Approximate Optimization Algorithm (QAOA) implementation using PennyLane.

This module provides a comprehensive implementation of QAOA for solving
combinatorial optimization problems.
"""

import pennylane as qml
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import time
from scipy.optimize import minimize


@dataclass
class QAOAResult:
    """Container for QAOA optimization results."""
    optimal_params: np.ndarray
    best_cost: float
    cost_history: List[float]
    param_history: List[np.ndarray]
    optimization_time: float
    function_evaluations: int
    success: bool
    message: str
    final_state: Optional[np.ndarray] = None
    expectation_values: Optional[Dict[str, float]] = None


class QAOA:
    """
    Quantum Approximate Optimization Algorithm (QAOA) implementation.
    
    QAOA is a hybrid quantum-classical algorithm for solving combinatorial
    optimization problems. It uses a parameterized quantum circuit to prepare
    quantum states and a classical optimizer to find optimal parameters.
    
    Args:
        cost_hamiltonian: The cost Hamiltonian representing the optimization problem
        mixer_hamiltonian: The mixer Hamiltonian (default: X mixer)
        device: PennyLane quantum device
        layers: Number of QAOA layers (p parameter)
        optimizer: Classical optimizer method
        optimizer_options: Options for the classical optimizer
        shots: Number of shots for quantum measurements
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        cost_hamiltonian: qml.Hamiltonian,
        mixer_hamiltonian: Optional[qml.Hamiltonian] = None,
        device: Optional[qml.Device] = None,
        layers: int = 1,
        optimizer: str = "COBYLA",
        optimizer_options: Optional[Dict[str, Any]] = None,
        shots: int = 1000,
        seed: Optional[int] = None,
    ):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_qubits = len(cost_hamiltonian.wires)
        
        # Default mixer is X on all qubits
        if mixer_hamiltonian is None:
            coeffs = [1.0] * self.n_qubits
            ops = [qml.PauliX(i) for i in range(self.n_qubits)]
            self.mixer_hamiltonian = qml.Hamiltonian(coeffs, ops)
        else:
            self.mixer_hamiltonian = mixer_hamiltonian
            
        # Set up device
        if device is None:
            self.device = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        else:
            self.device = device
            
        self.layers = layers
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options or {}
        self.shots = shots
        self.seed = seed
        
        # Initialize random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Optimization tracking
        self.cost_history = []
        self.param_history = []
        self.function_evaluations = 0
        
        # Create quantum circuit
        self._create_circuit()
    
    def _create_circuit(self):
        """Create the QAOA quantum circuit."""
        
        @qml.qnode(self.device)
        def qaoa_circuit(params):
            # Initialize in |+⟩ state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply QAOA layers
            for layer in range(self.layers):
                # Cost layer: exp(-i * gamma * H_C)
                gamma = params[layer]
                qml.ApproxTimeEvolution(self.cost_hamiltonian, gamma, 1)
                
                # Mixer layer: exp(-i * beta * H_M)
                beta = params[layer + self.layers]
                qml.ApproxTimeEvolution(self.mixer_hamiltonian, beta, 1)
            
            return qml.expval(self.cost_hamiltonian)
        
        self.circuit = qaoa_circuit
    
    def cost_function(self, params: np.ndarray) -> float:
        """
        Evaluate the cost function for given parameters.
        
        Args:
            params: QAOA parameters [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            
        Returns:
            Cost function value (expectation value of cost Hamiltonian)
        """
        self.function_evaluations += 1
        cost = self.circuit(params)
        
        # Track optimization progress
        self.cost_history.append(float(cost))
        self.param_history.append(params.copy())
        
        return float(cost)
    
    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        callback: Optional[Callable] = None,
    ) -> QAOAResult:
        """
        Optimize QAOA parameters.
        
        Args:
            initial_params: Initial parameter values
            max_iterations: Maximum optimization iterations
            callback: Optional callback function
            
        Returns:
            QAOAResult containing optimization results
        """
        start_time = time.time()
        
        # Initialize parameters
        if initial_params is None:
            # Random initialization
            initial_params = np.random.uniform(0, 2*np.pi, 2*self.layers)
        
        # Reset tracking
        self.cost_history = []
        self.param_history = []
        self.function_evaluations = 0
        
        # Set up optimizer options
        options = {
            "maxiter": max_iterations,
            "disp": False,
            **self.optimizer_options
        }
        
        # Custom callback that includes user callback
        def combined_callback(xk):
            if callback:
                callback(xk)
        
        # Run optimization
        try:
            result = minimize(
                self.cost_function,
                initial_params,
                method=self.optimizer,
                options=options,
                callback=combined_callback if callback else None,
            )
            
            optimization_time = time.time() - start_time
            
            # Get final quantum state if possible
            final_state = None
            expectation_values = None
            
            if hasattr(self.device, 'state'):
                # For state vector simulators
                @qml.qnode(self.device)
                def state_circuit(params):
                    # Initialize in |+⟩ state
                    for i in range(self.n_qubits):
                        qml.Hadamard(wires=i)
                    
                    # Apply QAOA layers
                    for layer in range(self.layers):
                        gamma = params[layer]
                        qml.ApproxTimeEvolution(self.cost_hamiltonian, gamma, 1)
                        
                        beta = params[layer + self.layers]
                        qml.ApproxTimeEvolution(self.mixer_hamiltonian, beta, 1)
                    
                    return qml.state()
                
                final_state = state_circuit(result.x)
                
                # Calculate additional expectation values
                expectation_values = {
                    "cost_hamiltonian": float(self.circuit(result.x)),
                    "mixer_hamiltonian": float(qml.ExpvalCost(state_circuit, self.mixer_hamiltonian, self.device)(result.x))
                }
            
            return QAOAResult(
                optimal_params=result.x,
                best_cost=result.fun,
                cost_history=self.cost_history.copy(),
                param_history=self.param_history.copy(),
                optimization_time=optimization_time,
                function_evaluations=self.function_evaluations,
                success=result.success,
                message=result.message,
                final_state=final_state,
                expectation_values=expectation_values,
            )
            
        except Exception as e:
            optimization_time = time.time() - start_time
            
            return QAOAResult(
                optimal_params=initial_params,
                best_cost=float('inf'),
                cost_history=self.cost_history.copy(),
                param_history=self.param_history.copy(),
                optimization_time=optimization_time,
                function_evaluations=self.function_evaluations,
                success=False,
                message=str(e),
            )
    
    def sample_bitstrings(
        self,
        params: np.ndarray,
        n_samples: int = 1000,
    ) -> Tuple[List[str], List[float]]:
        """
        Sample bitstrings from the QAOA state.
        
        Args:
            params: QAOA parameters
            n_samples: Number of samples to draw
            
        Returns:
            Tuple of (bitstrings, probabilities)
        """
        @qml.qnode(self.device)
        def sampling_circuit(params):
            # Initialize in |+⟩ state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply QAOA layers
            for layer in range(self.layers):
                gamma = params[layer]
                qml.ApproxTimeEvolution(self.cost_hamiltonian, gamma, 1)
                
                beta = params[layer + self.layers]
                qml.ApproxTimeEvolution(self.mixer_hamiltonian, beta, 1)
            
            return qml.sample(wires=range(self.n_qubits))
        
        # Get samples
        samples = []
        for _ in range(n_samples):
            sample = sampling_circuit(params)
            bitstring = ''.join(map(str, sample))
            samples.append(bitstring)
        
        # Count occurrences
        unique_bitstrings, counts = np.unique(samples, return_counts=True)
        probabilities = counts / n_samples
        
        return list(unique_bitstrings), list(probabilities)
    
    def get_quantum_circuit_info(self) -> Dict[str, Any]:
        """Get information about the quantum circuit."""
        # Create a sample circuit to analyze
        params = np.zeros(2 * self.layers)
        
        @qml.qnode(self.device)
        def analysis_circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.layers):
                gamma = params[layer]
                qml.ApproxTimeEvolution(self.cost_hamiltonian, gamma, 1)
                
                beta = params[layer + self.layers]
                qml.ApproxTimeEvolution(self.mixer_hamiltonian, beta, 1)
            
            return qml.expval(self.cost_hamiltonian)
        
        # Get circuit representation
        analysis_circuit(params)
        
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.layers,
            "n_parameters": 2 * self.layers,
            "device": str(self.device),
            "cost_hamiltonian_terms": len(self.cost_hamiltonian.ops),
            "mixer_hamiltonian_terms": len(self.mixer_hamiltonian.ops),
        }