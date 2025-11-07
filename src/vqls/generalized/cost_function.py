"""
Cost function computation for DF-VQLS

Computes the global cost function CG(θ) using swap test circuits.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from .circuit_builder import CircuitBuilder
from .state_preparer import StatePreparer
from .ansatz import Ansatz


def extract_P0_from_statevector(statevector: np.ndarray) -> float:
    """
    Extract probability of measuring ancilla in |0⟩ state.

    For a statevector, P(0) is the sum of probabilities of all even-indexed
    basis states (where ancilla qubit is in |0⟩ state).

    Args:
        statevector: Quantum statevector from simulation

    Returns:
        Probability P(0) of measuring ancilla in |0⟩ state
    """
    P0 = 0.0
    for i in range(len(statevector)):
        if i % 2 == 0:  # Even indices correspond to ancilla = |0⟩
            P0 += abs(statevector[i]) ** 2
    return P0


def extract_P0_from_measurement(self, circuit: QuantumCircuit) -> float:
    """
    Extract probability of measuring ancilla in |0⟩ state from measurement counts.

    Args:
        circuit: Quantum circuit with measurements

    Returns:
        Probability P(0) of measuring ancilla in |0⟩ state
    """
    num_qubits = circuit.num_qubits
    measured_circ = QuantumCircuit(num_qubits, 1)  # 1 classical bit
    measured_circ.compose(circuit, inplace=True)  # Copy gates from original
    measured_circ.measure(0, 0)  # Measure ancilla qubit

    # Run with shots
    transpiled = transpile(measured_circ, self.simulator)
    result = self.simulator.run(transpiled, shots=self.shots).result()
    counts = result.get_counts()

    # Extract P(0) - counts format: {'0': n0, '1': n1}
    count_0 = counts.get("0", 0)
    total_shots = sum(counts.values())
    P0 = count_0 / total_shots

    return P0


class CostFunction:
    """
    Computes the global cost function CG(θ) for DF-VQLS.

    Cost function: CG(θ) = 1 - |⟨f|K|u(θ)⟩|² / ⟨u(θ)|K^T K|u(θ)⟩

    When CG(θ) = 0, we have found the exact solution.
    """

    def __init__(
        self,
        circuit_builder: CircuitBuilder,
        state_preparer: StatePreparer,
        ansatz: Ansatz,
        simulator: AerSimulator,
        shots: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize cost function.

        Args:
            circuit_builder: Circuit builder instance
            state_preparer: State preparer instance
            ansatz: Ansatz instance
            simulator: Qiskit Aer simulator
            verbose: Whether to print detailed cost information
        """
        self.circuit_builder = circuit_builder
        self.state_preparer = state_preparer
        self.ansatz = ansatz
        self.simulator = simulator
        self.verbose = verbose

        self.n_qubits = circuit_builder.n_qubits

        # Pre-created simulators for parallel execution (lazy initialization)
        self._parallel_simulators = None
        self._parallel_executor = None
        self.shots = shots

    def _compute_u_theta(self, params: np.ndarray) -> np.ndarray:
        """
        Compute |u(θ)⟩ by running the ansatz.

        Args:
            params: Ansatz parameters

        Returns:
            Quantum state vector |u(θ)⟩
        """
        # Create temporary circuit for ansatz
        temp_circ = QuantumCircuit(self.n_qubits)
        qubits = list(range(self.n_qubits))
        temp_circ = self.ansatz.apply(temp_circ, qubits, params)
        temp_circ.save_statevector()

        # Run simulation
        transpiled_circ = transpile(temp_circ, self.simulator)
        result = self.simulator.run(transpiled_circ).result()
        u_theta = np.asarray(result.get_statevector(temp_circ))

        return u_theta

    def _compute_u_theta_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Build the ansatz circuit |u(θ)⟩ with parameters bound.
        """
        temp_circ = QuantumCircuit(self.n_qubits)
        qubits = list(range(self.n_qubits))
        temp_circ = self.ansatz.apply(temp_circ, qubits, params)
        temp_circ.name = "U(θ)"
        return temp_circ

    def _initialize_parallel_resources(self, num_workers: int = 2):
        """Initialize parallel execution resources (simulators and executor).

        This pre-creates simulator instances to avoid initialization overhead
        during parallel execution.
        """
        if self._parallel_simulators is None:
            from qiskit_aer import AerSimulator

            # Pre-create simulator instances for each worker
            self._parallel_simulators = [
                AerSimulator(method="statevector") for _ in range(num_workers)
            ]
            # Create thread pool executor (reuse it)
            self._parallel_executor = ThreadPoolExecutor(max_workers=num_workers)

    def _run_circuit_parallel(self, circuits: list) -> list:
        """Run multiple circuits in parallel.

        Uses pre-initialized simulator instances to minimize overhead.
        Each thread gets its own simulator instance since AerSimulator is not thread-safe.

        Note: Assumes _initialize_parallel_resources() has already been called.

        Args:
            circuits: List of circuits to run in parallel
        """
        num_workers = len(circuits)

        # Ensure parallel resources are initialized
        if (
            self._parallel_simulators is None
            or len(self._parallel_simulators) < num_workers
        ):
            self._initialize_parallel_resources(num_workers)

        def run_single_with_simulator(circ_and_sim):
            """Run circuit with pre-created simulator."""
            circ, sim = circ_and_sim
            transpiled = transpile(circ, sim)
            result = sim.run(transpiled).result()
            return np.asarray(result.get_statevector(circ))

        # Pair each circuit with a pre-created simulator
        circuits_with_sims = list(
            zip(circuits, self._parallel_simulators[:num_workers])
        )

        # Execute in parallel using pre-created executor
        from concurrent.futures import as_completed

        # Use submit to get futures
        futures = [
            self._parallel_executor.submit(run_single_with_simulator, circ_and_sim)
            for circ_and_sim in circuits_with_sims
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            pass  # Just wait for completion

        # Get results in original order (preserve circuit order)
        ordered_results = [future.result() for future in futures]

        return ordered_results

    def extract_P0_from_measurement(self, circuit: QuantumCircuit) -> float:
        """
        Extract probability of measuring ancilla in |0⟩ state from measurement counts.

        Args:
            circuit: Quantum circuit with measurements

        Returns:
            Probability P(0) of measuring ancilla in |0⟩ state
        """
        num_qubits = circuit.num_qubits
        measured_circ = QuantumCircuit(num_qubits, 1)  # 1 classical bit
        measured_circ.compose(circuit, inplace=True)  # Copy gates from original
        measured_circ.measure(0, 0)  # Measure ancilla qubit

        # Run with shots
        transpiled = transpile(measured_circ, self.simulator)
        result = self.simulator.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

        # Extract P(0) - counts format: {'0': n0, '1': n1}
        count_0 = counts.get("0", 0)
        total_shots = sum(counts.values())
        P0 = count_0 / total_shots

        return P0

    def compute(
        self,
        params: np.ndarray,
        K: np.ndarray,
        f: np.ndarray,
        pbar: Optional[object] = None,
        use_parallel: bool = False,  # Disable by default - enable when needed
        phase_pbar: Optional[object] = None,  # Progress bar for internal phases
    ) -> float:
        """
        Compute the global cost function CG(θ).

        Args:
            params: Ansatz parameters
            K: Coefficient matrix
            f: Right-hand side vector
            pbar: Optional tqdm progress bar for updating
            use_parallel: Whether to run numerator and denominator circuits in parallel
            phase_pbar: Optional progress bar for internal phases

        Returns:
            Cost function value (0 means perfect solution)
        """
        from tqdm import tqdm

        # Create phase progress bar if not provided and verbose mode
        if phase_pbar is None and self.verbose:
            phase_pbar = tqdm(
                total=6,
                desc="Cost computation",
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}",
            )

        # Phase 1: Prepare states (cached - only computed once if K and f don't change)
        if phase_pbar:
            phase_pbar.set_description("Phase 1/6: Preparing states")
        vec_K, norm_K = self.state_preparer.prepare_matrix(K)
        vec_KT, _ = self.state_preparer.prepare_matrix_transpose(K)
        f_norm = self.state_preparer.prepare_vector(f)
        if phase_pbar:
            phase_pbar.update(1)

        if self.shots is None:
            # === STATEVECTOR (NOISELESS) PATH ===
            # Phase 2: Compute |u(θ)⟩ as statevector
            if phase_pbar:
                phase_pbar.set_description("Phase 2/6: Computing |u(θ)⟩ (fast)")
            u_theta_array = self._compute_u_theta(params)  # <-- Use fast array func
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 3: Build circuits (using initialize)
            if phase_pbar:
                phase_pbar.set_description("Phase 3/6: Building circuits (fast)")
            circ_num = self.circuit_builder.build_numerator_circuit(
                vec_K, f_norm, u_theta_array
            )
            circ_den = self.circuit_builder.build_denominator_circuit(
                vec_K, vec_KT, u_theta_array
            )
            circ_num.save_statevector()
            circ_den.save_statevector()
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 4: Run circuits (parallel or sequential)
            if phase_pbar:
                phase_pbar.set_description("Phase 4/6: Running circuits (Statevector)")

            # Phase 4: Run numerator circuit (sequential)
            if phase_pbar:
                phase_pbar.set_description("phase 4/6: running numerator (statevector)")
            transpiled_num = transpile(circ_num, self.simulator)
            result_num = self.simulator.run(transpiled_num).result()
            sv_num = np.asarray(result_num.get_statevector(circ_num))
            if phase_pbar:
                phase_pbar.update(1)

            # phase 5: run denominator circuit
            if phase_pbar:
                phase_pbar.set_description(
                    "phase 5/6: running denominator (statevector)"
                )
            transpiled_den = transpile(circ_den, self.simulator)
            result_den = self.simulator.run(transpiled_den).result()
            sv_den = np.asarray(result_den.get_statevector(circ_den))
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 6a: Compute P(0) from statevectors
            if phase_pbar:
                phase_pbar.set_description("Phase 6/6: Computing cost")
            P0_num = extract_P0_from_statevector(sv_num)
            P0_den = extract_P0_from_statevector(sv_den)

        else:
            # === SHOT-BASED (NOISY) PATH ===

            # Phase 2: Build |u(θ)⟩ circuit
            if phase_pbar:
                phase_pbar.set_description("Phase 2/6: Building |u(θ)⟩ circuit")
            u_theta_circuit = self._compute_u_theta_circuit(
                params
            )  # <-- Use circuit func
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 3: Build circuits (using StatePreparation)
            if phase_pbar:
                phase_pbar.set_description("Phase 3/6: Building circuits (quantum)")
            circ_num = self.circuit_builder.build_numerator_circuit_quantum(
                vec_K, f_norm, u_theta_circuit
            )
            circ_den = self.circuit_builder.build_denominator_circuit_quantum(
                vec_K, vec_KT, u_theta_circuit
            )
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 4/5: Run circuits on NOISY simulator (self.simulator)
            if phase_pbar:
                phase_pbar.set_description("Phase 4/6: Running numerator (Shots)")
            P0_num = self.extract_P0_from_measurement(circ_num)
            if phase_pbar:
                phase_pbar.update(1)

            if phase_pbar:
                phase_pbar.set_description("Phase 5/6: Running denominator (Shots)")
            P0_den = self.extract_P0_from_measurement(circ_den)
            if phase_pbar:
                phase_pbar.update(1)

            # Phase 6a: P(0) values are already from counts
            if phase_pbar:
                phase_pbar.set_description("Phase 6/6: Computing cost")

        # Calculate inner product squared: |⟨ψ₁|ψ₂⟩|² = 2P(0) - 1
        inner_product_squared_num = abs(2 * P0_num - 1)
        inner_product_squared_den = abs(2 * P0_den - 1)

        # Numerator needs |⟨vec(K)|u,f⟩|² (squared overlap)
        numerator = (norm_K**2) * inner_product_squared_num

        # Denominator needs ⟨u|K^T K|u⟩ = ⟨u⊗vec(K^T)|vec(K)⊗u⟩ (inner product, NOT squared!)
        # Swap test gives |⟨ψ₁|ψ₂⟩|², so take sqrt to get the inner product value
        inner_product_den = np.sqrt(inner_product_squared_den)
        denominator = (norm_K**2) * inner_product_den

        # Compute cost function
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        cost = 1.0 - (numerator / (denominator + epsilon))

        if phase_pbar:
            phase_pbar.update(1)
            phase_pbar.close()

        # Update progress bar if provided
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"cost": f"{cost:.6f}"})

        # Print detailed information if verbose
        if self.verbose and pbar is None:
            print(
                f"Cost: {cost:.6f} | Num: {numerator:.6f} | "
                f"Den: {denominator:.6f} | P0_num: {P0_num:.4f} | "
                f"P0_den: {P0_den:.4f}"
            )

        return cost
