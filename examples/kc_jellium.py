import openfermion
import openfermioncirq

# Set parameters of jellium model.
wigner_seitz_radius = 5. # Radius per electron in Bohr radii.
n_dimensions = 2 # Number of spatial dimensions.
grid_length = 3 # Number of grid points in each dimension.
spinless = True # Whether to include spin degree of freedom or not.
n_electrons = 2 # Number of electrons.

# Figure out length scale based on Wigner-Seitz radius and construct a basis grid.
length_scale = openfermion.wigner_seitz_length_scale(
    wigner_seitz_radius, n_electrons, n_dimensions)
grid = openfermion.Grid(n_dimensions, grid_length, length_scale)

# Initialize the model and compute its ground energy in the correct particle number manifold
fermion_hamiltonian = openfermion.jellium_model(grid, spinless=spinless, plane_wave=False)
hamiltonian_sparse = openfermion.get_sparse_operator(fermion_hamiltonian)
ground_energy, _ = openfermion.jw_get_ground_state_at_particle_number(
    hamiltonian_sparse, n_electrons)
print('The ground energy of the jellium Hamiltonian at {} electrons is {}'.format(
    n_electrons, ground_energy))

# Convert to DiagonalCoulombHamiltonian type.
hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(fermion_hamiltonian)

# Define the objective function
objective = openfermioncirq.HamiltonianObjective(hamiltonian)

# Create a swap network Trotter ansatz.
iterations = 1  # This is the number of Trotter steps to use in the ansatz.
ansatz = openfermioncirq.SwapNetworkTrotterAnsatz(
    hamiltonian,
    iterations=iterations)

print('Created a variational ansatz with the following circuit:')
print(ansatz.circuit.to_text_diagram(transpose=True))

# Use preparation circuit for mean-field state
import cirq
preparation_circuit = cirq.Circuit(
    openfermioncirq.prepare_gaussian_state(
        ansatz.qubits,
        openfermion.QuadraticHamiltonian(hamiltonian.one_body),
        occupied_orbitals=range(n_electrons)))

kc_simulator = cirq.KnowledgeCompilationSimulator(
    preparation_circuit + ansatz.circuit,
    initial_state=0
)

# Create a Hamiltonian variational study
study = openfermioncirq.VariationalStudy(
    'jellium_study',
    ansatz,
    objective,
    preparation_circuit=preparation_circuit)

print("Created a variational study with {} qubits and {} parameters".format(
    len(study.ansatz.qubits), study.num_params))

print("The value of the objective with default initial parameters is {}".format(
    study.value_of(kc_simulator, ansatz.default_initial_params())))

print("The circuit of the study is")
print(study.circuit.to_text_diagram(transpose=True))

# Perform an optimization run.
from openfermioncirq.optimization import ScipyOptimizationAlgorithm, OptimizationParams
algorithm = ScipyOptimizationAlgorithm(
    kwargs={'method': 'COBYLA'},
    options={'maxiter': 100},
    uses_bounds=False)
optimization_params = OptimizationParams(
    algorithm=algorithm)
result = study.optimize(optimization_params, kc_simulator)
print(result.optimal_value)

optimization_params = OptimizationParams(
    algorithm=algorithm,
    cost_of_evaluate=1e6)
study.optimize(
    optimization_params,
    kc_simulator,
    identifier='COBYLA with maxiter=100, noisy',
    repetitions=3,
    reevaluate_final_params=True,
    use_multiprocessing=False)
print(study)
