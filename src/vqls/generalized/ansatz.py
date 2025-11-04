"""
Ansatz definitions for DF-VQLS

Provides hardware-efficient ansatz and interface for future ansatz types.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from qiskit import QuantumCircuit


class Ansatz(ABC):
    """Abstract base class for variational ansatz."""
    
    def __init__(self, num_qubits: int, num_layers: int = 3):
        """
        Initialize ansatz.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of ansatz layers
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
    
    @abstractmethod
    def apply(
        self,
        circ: QuantumCircuit,
        qubits: List[int],
        parameters: np.ndarray
    ) -> QuantumCircuit:
        """
        Apply ansatz to quantum circuit.
        
        Args:
            circ: Quantum circuit to apply ansatz to
            qubits: List of qubit indices
            parameters: Flattened parameter array
        
        Returns:
            Modified quantum circuit with ansatz applied
        """
        pass
    
    def num_parameters(self) -> int:
        """Calculate number of parameters needed."""
        return self.num_qubits * self.num_layers


class HardwareEfficientAnsatz(Ansatz):
    """
    Hardware-efficient ansatz with RY rotations and CZ entanglement.
    
    Each layer consists of:
    - Parameterized RY rotation on each qubit
    - CZ entanglement gates in a specific pattern
    
    The entanglement pattern:
    - Linear: q0-q1, q1-q2, ..., q_{n-2}-q_{n-1}
    - Circular: q_{n-1}-q0 (if num_qubits > 2)
    """
    
    def apply(
        self,
        circ: QuantumCircuit,
        qubits: List[int],
        parameters: np.ndarray
    ) -> QuantumCircuit:
        """
        Apply hardware-efficient ansatz.
        
        Args:
            circ: Quantum circuit to apply ansatz to
            qubits: List of qubit indices
            parameters: Flattened parameter array (num_qubits × num_layers)
        
        Returns:
            Modified quantum circuit with ansatz applied
        """
        num_qubits = len(qubits)
        params_reshaped = parameters.reshape(self.num_layers, num_qubits)
        
        for layer in range(self.num_layers):
            # RY rotations on each qubit
            for i, qubit in enumerate(qubits):
                circ.ry(params_reshaped[layer, i], qubit)
            
            # CZ entanglement (skip on last layer)
            if layer < self.num_layers - 1:
                # Linear entanglement: q0-q1, q1-q2, ..., q_{n-2}-q_{n-1}
                for i in range(num_qubits - 1):
                    circ.cz(qubits[i], qubits[i + 1])
                
                # Circular entanglement: q_{n-1}-q0 (if more than 2 qubits)
                if num_qubits > 2:
                    circ.cz(qubits[num_qubits - 1], qubits[0])
        
        return circ

