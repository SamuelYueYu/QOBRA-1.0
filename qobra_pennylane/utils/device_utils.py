"""
Device utilities for creating and managing PennyLane devices.

This module provides helper functions for creating quantum devices
with appropriate configurations for different backends.
"""

import pennylane as qml
from typing import Optional, Dict, Any, Union


def create_device(
    device_name: str = "default.qubit",
    wires: Optional[Union[int, list]] = None,
    shots: Optional[int] = None,
    **kwargs
) -> qml.Device:
    """
    Create a PennyLane quantum device with standard configurations.
    
    Args:
        device_name: Name of the device (e.g., 'default.qubit', 'lightning.qubit')
        wires: Number of wires or list of wire labels
        shots: Number of shots for finite sampling (None for exact)
        **kwargs: Additional device-specific parameters
        
    Returns:
        Configured PennyLane device
    """
    device_kwargs = {}
    
    if wires is not None:
        device_kwargs['wires'] = wires
    
    if shots is not None:
        device_kwargs['shots'] = shots
    
    # Add any additional kwargs
    device_kwargs.update(kwargs)
    
    try:
        device = qml.device(device_name, **device_kwargs)
        return device
    except Exception as e:
        raise ValueError(f"Failed to create device '{device_name}': {e}")


def get_available_devices() -> Dict[str, bool]:
    """
    Get a dictionary of available PennyLane devices.
    
    Returns:
        Dictionary mapping device names to availability status
    """
    common_devices = [
        "default.qubit",
        "default.mixed", 
        "lightning.qubit",
        "qulacs.simulator",
        "cirq.simulator",
        "qiskit.aer",
        "qiskit.ibmq",
        "qiskit.basicaer",
    ]
    
    available = {}
    
    for device_name in common_devices:
        try:
            # Try to create a minimal device to test availability
            qml.device(device_name, wires=1)
            available[device_name] = True
        except:
            available[device_name] = False
    
    return available


def recommend_device(
    n_qubits: int,
    shots: Optional[int] = None,
    prefer_exact: bool = True,
) -> str:
    """
    Recommend an appropriate device for the given problem size.
    
    Args:
        n_qubits: Number of qubits needed
        shots: Number of shots (None for exact computation)
        prefer_exact: Whether to prefer exact computation when possible
        
    Returns:
        Recommended device name
    """
    available = get_available_devices()
    
    # For small problems, prefer exact simulation
    if n_qubits <= 12 and prefer_exact and shots is None:
        if available.get("lightning.qubit", False):
            return "lightning.qubit"
        elif available.get("default.qubit", False):
            return "default.qubit"
    
    # For larger problems or shot-based simulation
    if available.get("lightning.qubit", False):
        return "lightning.qubit"
    elif available.get("default.qubit", False):
        return "default.qubit"
    else:
        raise RuntimeError("No suitable quantum device available")


def configure_device_for_problem(
    problem_size: int,
    algorithm: str = "qaoa",
    shots: Optional[int] = 1000,
    **kwargs
) -> qml.Device:
    """
    Configure an appropriate device for a specific problem and algorithm.
    
    Args:
        problem_size: Size of the optimization problem (number of variables/qubits)
        algorithm: Algorithm to be used ('qaoa', 'vqe', etc.)
        shots: Number of shots for sampling
        **kwargs: Additional device configuration options
        
    Returns:
        Configured PennyLane device
    """
    # Determine appropriate device
    if algorithm.lower() in ["qaoa", "vqe"]:
        if shots is None:
            # Exact computation
            device_name = recommend_device(problem_size, shots=None, prefer_exact=True)
        else:
            # Shot-based computation
            device_name = recommend_device(problem_size, shots=shots, prefer_exact=False)
    else:
        # Default recommendation
        device_name = recommend_device(problem_size, shots=shots)
    
    # Create device
    device = create_device(
        device_name=device_name,
        wires=problem_size,
        shots=shots,
        **kwargs
    )
    
    return device