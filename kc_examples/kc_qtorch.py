"""Creates and simulates a simple circuit.
"""

import cirq
import qsimcirq
import os
import re

import numpy as np
import math

def main():

    q0,q1,q2 = cirq.LineQubit.range(3)
    for iteration in range(1):
        random_circuit = cirq.testing.random_circuit(qubits=[q0,q1,q2],
                                                     n_moments=32,
                                                     op_density=0.99)

        cirq.ConvertToCzAndSingleGates().optimize_circuit(random_circuit) # cannot work with params
        cirq.ExpandComposite().optimize_circuit(random_circuit)
        qs_circuit = qsimcirq.QSimCircuit(random_circuit)
        random_circuit._to_quil_output().save_to_file('kc_qtorch.quil')

        qs_simulator = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 2})
        qs_result = qs_simulator.simulate(qs_circuit)
        assert qs_result.state_vector().shape == (8,)
        kc_simulator = cirq.KnowledgeCompilationSimulator(random_circuit)
        kc_result = kc_simulator.simulate(random_circuit)
        print("qs_result.state_vector()")
        print(qs_result.state_vector())
        print("kc_result.state_vector()")
        print(kc_result.state_vector())
        assert cirq.linalg.allclose_up_to_global_phase(
            qs_result.state_vector(),
            kc_result.state_vector(),
            rtol = 1.e-4,
            atol = 1.e-6,
        )

        path_to_qtorch = '/common/home/yh804/research/qtorch/bin/qtorch'
        with open('kc_qtorch.inp','w') as inp_file:
            inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch.quil
# >string contractmethod simple-stoch
>string contractmethod linegraph-qbb
>int quickbbseconds 65536
# >string measurement kc_qtorch.meas
# >string outputpath kc_qtorch.out
>bool qbbonly true
# >bool readqbbresonly true
# >int threads 8
''')
        stdout = os.system(path_to_qtorch + ' kc_qtorch.inp')

        probs=np.zeros(1<<3)
        for bitstring in range(1<<3):
            with open('kc_qtorch.inp','w') as inp_file:
                inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch.quil
# >string contractmethod simple-stoch
>string contractmethod linegraph-qbb
# >int quickbbseconds 65536
>string measurement kc_qtorch.meas
>string outputpath kc_qtorch.out
# >bool qbbonly true
>bool readqbbresonly true
>int threads 8
''')
            with open('kc_qtorch.meas','w') as meas_file:
                meas_file.write("{:03b}".format(bitstring))
            stdout = os.system(path_to_qtorch + ' kc_qtorch.inp')
            with open('kc_qtorch.out','r') as out_file:
                for line in out_file.readlines():
                    words = re.split(r'\(|,',line)
                    if words[0]=='Result of Contraction: ':
                        probs[bitstring] = float(words[1])

        print("np.diag(np.outer(qs_result.state_vector(),np.conj(qs_result.state_vector())))")
        print(np.diag(np.outer(qs_result.state_vector(),np.conj(qs_result.state_vector()))))
        print("probs")
        print(probs)
        assert cirq.linalg.allclose_up_to_global_phase(
            np.diag(np.outer(qs_result.state_vector(),np.conj(qs_result.state_vector()))),
            probs,
            rtol = 1.e-4,
            atol = 1.e-6,
        )

        circuit_unitary = []
        for x in range(8):
            result = kc_simulator.simulate(random_circuit,
                                        initial_state=x)
            circuit_unitary.append(result.final_state_vector)

        print ("np.transpose(circuit_unitary) = ")
        print (np.transpose(circuit_unitary))
        print ("random_circuit.unitary() = ")
        print (random_circuit.unitary())
        np.testing.assert_almost_equal(
            np.transpose(circuit_unitary),
            random_circuit.unitary(),
            decimal=4)

if __name__ == '__main__':
    main()
