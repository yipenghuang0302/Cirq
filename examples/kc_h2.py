import openfermion

diatomic_bond_length = .7414
geometry = [('H', (0., 0., 0.)),
            ('H', (0., 0., diatomic_bond_length))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
description = format(diatomic_bond_length)

molecule = openfermion.MolecularData(
    geometry,
    basis,
    multiplicity,
    description=description)
molecule.load()

hamiltonian = molecule.get_molecular_hamiltonian()
print("Bond Length in Angstroms: {}".format(diatomic_bond_length))
print("Hartree Fock (mean-field) energy in Hartrees: {}".format(molecule.hf_energy))
print("FCI (Exact) energy in Hartrees: {}".format(molecule.fci_energy))

import cirq
import openfermioncirq
import sympy

class MyAnsatz(openfermioncirq.VariationalAnsatz):

    def params(self):
        """The parameters of the ansatz."""
        return [sympy.Symbol('theta_0')]

    def operations(self, qubits):
        """Produce the operations of the ansatz circuit."""
        q0, q1, q2, q3 = qubits
        yield cirq.H(q0), cirq.H(q1), cirq.H(q2)
        yield cirq.XPowGate(exponent=-0.5).on(q3)

        yield cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)
        yield cirq.ZPowGate(exponent=sympy.Symbol('theta_0')).on(q3)
        yield cirq.CNOT(q2, q3), cirq.CNOT(q1, q2), cirq.CNOT(q0, q1)

        yield cirq.H(q0), cirq.H(q1), cirq.H(q2)
        yield cirq.XPowGate(exponent=0.5).on(q3)

        # yield cirq.measure(*qubits, key='m')

    def _generate_qubits(self):
        """Produce qubits that can be used by the ansatz circuit."""
        return cirq.LineQubit.range(4)

ansatz = MyAnsatz()
objective = openfermioncirq.HamiltonianObjective(hamiltonian)
q0, q1, _, _ = ansatz.qubits
preparation_circuit = cirq.Circuit(
    cirq.X(q0),
    cirq.X(q1))
kc_simulator = cirq.KnowledgeCompilationSimulator(
    preparation_circuit + ansatz.circuit,
    initial_state=0
)

study = openfermioncirq.VariationalStudy(
    name='my_hydrogen_study',
    ansatz=ansatz,
    objective=objective,
    preparation_circuit=preparation_circuit)
print(study.circuit)

# Perform optimization.
import numpy
from openfermioncirq.optimization import COBYLA, OptimizationParams
optimization_params = OptimizationParams(
    algorithm=COBYLA,
    initial_guess=[0.01],
    cost_of_evaluate=4096)
result = study.optimize(optimization_params, kc_simulator)
print("Initial state energy in Hartrees: {}".format(molecule.hf_energy))
print("Optimized energy result in Hartree: {}".format(result.optimal_value))
print("Exact energy result in Hartees for reference: {}".format(molecule.fci_energy))

bond_lengths = ['{0:.1f}'.format(0.3 + 0.1 * x) for x in range(23)]
hartree_fock_energies = []
optimized_energies = []
exact_energies = []


for diatomic_bond_length in bond_lengths:
    geometry = [('H', (0., 0., 0.)),
                ('H', (0., 0., diatomic_bond_length))]

    description = format(diatomic_bond_length)

    molecule = openfermion.MolecularData(geometry, basis,
                                         multiplicity, description=description)
    molecule.load()
    hamiltonian = molecule.get_molecular_hamiltonian()

    study = openfermioncirq.VariationalStudy(
        name='my_hydrogen_study',
        ansatz=ansatz,
        objective=openfermioncirq.HamiltonianObjective(hamiltonian),
        preparation_circuit=preparation_circuit)

    result = study.optimize(optimization_params, kc_simulator=kc_simulator)
    hartree_fock_energies.append(molecule.hf_energy)
    optimized_energies.append(result.optimal_value)
    exact_energies.append(molecule.fci_energy)

    print("R={}\t Optimized Energy: {}".format(diatomic_bond_length, result.optimal_value))


import matplotlib
import matplotlib.pyplot as pyplot

# Plot the energy mean and std Dev
fig = pyplot.figure(figsize=(10,7))
bkcolor = '#ffffff'
ax = fig.add_subplot(1, 1, 1)
pyplot.subplots_adjust(left=.2)
ax.set_xlabel('R (Angstroms)')
ax.set_ylabel(r'E Hartrees')
ax.set_title(r'H$_2$ bond dissociation curve')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
bond_lengths = [float(x) for x in bond_lengths]
ax.plot(bond_lengths, hartree_fock_energies, label='Hartree-Fock')
ax.plot(bond_lengths, optimized_energies, '*', label='Optimized')
ax.plot(bond_lengths, exact_energies, '--', label='Exact')

ax.legend(frameon=False)
pyplot.show()
