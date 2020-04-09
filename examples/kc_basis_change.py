import numpy
import openfermion
import openfermioncirq
import cirq

# Set the number of qubits in our example.
n_qubits = 8
simulation_time = 1.
random_seed = 8317

# Generate the random one-body operator.
T = openfermion.random_hermitian_matrix(n_qubits, seed=random_seed)

# Diagonalize T and obtain basis transformation matrix (aka "u").
eigenvalues, eigenvectors = numpy.linalg.eigh(T)
basis_transformation_matrix = eigenvectors.transpose()

# Print out familiar OpenFermion "FermionOperator" form of H.
H = openfermion.FermionOperator()
for p in range(n_qubits):
    for q in range(n_qubits):
        term = ((p, 1), (q, 0))
        H += openfermion.FermionOperator(term, T[p, q])
print(H)

# Initialize the qubit register.
qubits = cirq.LineQubit.range(n_qubits)

# Start circuit with the inverse basis rotation, print out this step.
inverse_basis_rotation = cirq.inverse(openfermioncirq.bogoliubov_transform(qubits, basis_transformation_matrix))
circuit = cirq.Circuit(inverse_basis_rotation)
print(circuit)

# Add diagonal phase rotations to circuit.
for k, eigenvalue in enumerate(eigenvalues):
    phase = -eigenvalue * simulation_time
    circuit.append(cirq.Rz(rads=phase).on(qubits[k]))

# Finally, restore basis.
basis_rotation = openfermioncirq.bogoliubov_transform(qubits, basis_transformation_matrix)
circuit.append(basis_rotation)

print(circuit.to_text_diagram(transpose=True))

# Initialize a random initial state.
initial_state = openfermion.haar_random_vector(
    2 ** n_qubits, random_seed).astype(numpy.complex64)

# Numerically compute the correct circuit output.
import scipy
hamiltonian_sparse = openfermion.get_sparse_operator(H)
exact_state = scipy.sparse.linalg.expm_multiply(
    -1j * simulation_time * hamiltonian_sparse, initial_state)

# Use Cirq simulator to apply circuit.
simulator = cirq.Simulator()
result = simulator.simulate(circuit, qubit_order=qubits,
                            initial_state=initial_state)
simulated_state = result.final_state

# Print final fidelity.
fidelity = abs(numpy.dot(simulated_state, numpy.conjugate(exact_state)))**2
print(fidelity)
