# -*- coding: utf-8 -*-
"""Creates and simulates basic arithmetic circuits

=== EXAMPLE OUTPUT ===
Execute Adder
0: ───────────@───────────────────────────────────@───────────────@───
              │                                   │               │
1: ───@───@───┼───────────────────────────────────┼───@───@───@───┼───
      │   │   │                                   │   │   │   │   │
2: ───@───X───@───────────────────────────────────@───X───@───X───X───
      │       │                                   │       │
3: ───X───────X───────@───────@───────────────@───X───────X───────────
                      │       │               │
4: ───────────@───@───┼───────┼───@───@───@───┼───────────────────────
              │   │   │       │   │   │   │   │
5: ───────────@───X───@───────@───X───@───X───X───────────────────────
              │       │       │       │
6: ───────────X───────X───@───X───────X───────────────────────────────
                          │
7: ───────────────────@───┼───────────────────────────────────────────
                      │   │
8: ───────────────────X───X───────────────────────────────────────────
000 + 000 = 000
000 + 001 = 001
000 + 010 = 010
000 + 011 = 011
001 + 000 = 001
001 + 001 = 010
001 + 010 = 011
001 + 011 = 100
010 + 000 = 010
010 + 001 = 011
010 + 010 = 100
010 + 011 = 101
011 + 000 = 011
011 + 001 = 100
011 + 010 = 101
011 + 011 = 110

Execute Multiplier
0: ─────────────Adder:0───────────────────────Adder:0───────────────Adder:0─────
                │                             │                     │
1: ─X───────────Adder:1───X───────────────────Adder:1───────────────Adder:1─────
    │           │         │                   │                     │
2: ─┼───────────Adder:2───┼───────────────────Adder:2───────────────Adder:2─────
    │           │         │                   │                     │
3: ─┼───────────Adder:3───┼───────────────────Adder:3───────────────Adder:3─────
    │           │         │                   │                     │
4: ─┼───X───────Adder:4───┼───X───────X───────Adder:4───X───────────Adder:4─────
    │   │       │         │   │       │       │         │           │
5: ─┼───┼───────Adder:5───┼───┼───────┼───────Adder:5───┼───────────Adder:5─────
    │   │       │         │   │       │       │         │           │
6: ─┼───┼───────Adder:6───┼───┼───────┼───────Adder:6───┼───────────Adder:6─────
    │   │       │         │   │       │       │         │           │
7: ─┼───┼───X───Adder:7───┼───┼───X───┼───X───Adder:7───┼───X───X───Adder:7───X─
    │   │   │   │         │   │   │   │   │   │         │   │   │   │         │
8: ─┼───┼───┼───Adder:8───┼───┼───┼───┼───┼───Adder:8───┼───┼───┼───Adder:8───┼─
    │   │   │             │   │   │   │   │             │   │   │             │
9: ─@───┼───┼─────────────@───┼───┼───@───┼─────────────@───┼───@─────────────@─
    │   │   │             │   │   │   │   │             │   │   │             │
10:─┼───@───┼─────────────┼───@───┼───┼───@─────────────┼───@───┼─────────────┼─
    │   │   │             │   │   │   │   │             │   │   │             │
11:─┼───┼───@─────────────┼───┼───@───┼───┼─────────────┼───┼───┼─────────────┼─
    │   │   │             │   │   │   │   │             │   │   │             │
12:─@───@───@─────────────@───@───@───┼───┼─────────────┼───┼───┼─────────────┼─
                                      │   │             │   │   │             │
13:───────────────────────────────────@───@─────────────@───@───┼─────────────┼─
                                                                │             │
14:─────────────────────────────────────────────────────────────@─────────────@─
000 * 000 = 000
000 * 001 = 000
000 * 010 = 000
000 * 011 = 000
001 * 000 = 000
001 * 001 = 001
001 * 010 = 010
001 * 011 = 011
010 * 000 = 000
010 * 001 = 010
010 * 010 = 100
010 * 011 = 110
011 * 000 = 000
011 * 001 = 011
011 * 010 = 110
011 * 011 = 001
"""


import numpy as np
import sympy
import cirq


class Adder(cirq.Gate):
    """ A quantum circuit to calculate a + b

            -----------@---             ---@------------
                       |                   |
            ---@---@---+---             ---+---@---@---         -------@---
    [Carry]:   |   |   |      [Uncarry]:   |   |   |                   |
            ---@---X---@---             ---@---X---@---   [Sum]:---@---+---
               |       |                   |       |               |   |
            ---X-------X---             ---X-------X---         ---X---X---


           -----                                      -------    ---
    c0: --|     |------------------------------------|       |--|   |-----
          |     |                                    |       |  |   |
    a0: --|     |------------------------------------|       |--|Sum|-----
          |Carry|                                    |Uncarry|  |   |
    b0: --|     |------------------------------------|       |--|   |--M--
          |     |   -----           -------    ---   |       |   ---
    c1: --|     |--|     |---------|       |--|   |--|       |------------
           -----   |     |         |       |  |   |   -------
    a1: -----------|     |---------|       |--|Sum|-----------------------
                   |Carry|         |Uncarry|  |   |
    b1: -----------|     |---------|       |--|   |--------------------M--
                   |     |   ---   |       |   ---
    c2: -----------|     |--|   |--|       |------------------------------
                    -----   |   |   -------
    a2: --------------------|Sum|-----------------------------------------
                            |   |
    b2: --------------------|   |--------------------------------------M--
                             ---
    """

    def __init__(self, num_qubits):
        super(Adder, self)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def carry(self, *qubits):
        c0, a, b, c1 = qubits
        yield cirq.TOFFOLI(a, b, c1)
        yield cirq.CNOT(a, b)
        yield cirq.TOFFOLI(c0, b, c1)

    def uncarry(self, *qubits):
        c0, a, b, c1 = qubits
        yield cirq.TOFFOLI(c0, b, c1)
        yield cirq.CNOT(a, b)
        yield cirq.TOFFOLI(a, b, c1)

    def carry_sum(self, *qubits):
        c0, a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(c0, b)

    def _decompose_(self, qubits):
        n = int(len(qubits)/3)
        c = qubits[0::3]
        a = qubits[1::3]
        b = qubits[2::3]
        for i in range(n-1):
            yield self.carry(c[i], a[i], b[i], c[i+1])
        yield self.carry_sum(c[n-1], a[n-1], b[n-1])
        for i in range(n-2, -1, -1):
            yield self.uncarry(c[i], a[i], b[i], c[i+1])
            yield self.carry_sum(c[i], a[i], b[i])


class Multiplier(cirq.Gate):
    """ A quantum circuit to calculate y * x

                       -                         -                 -
    c0: --------------| |-----------------------| |---------------| |---------
                      | |                       | |               | |
    a0: --X-----------| |---X-------------------| |---------------| |---------
          |           | |   |                   | |               | |
    b0: --+-----------| |---+-------------------| |---------------| |------M--
          |           | |   |                   | |               | |
    c1: --+-----------|A|---+-------------------|A|---------------|A|---------
          |           |d|   |                   |d|               |d|
    a1: --+---X-------|d|---+---X-------X-------|d|---X-----------|d|---------
          |   |       |e|   |   |       |       |e|   |           |e|
    b1: --+---+-------|r|---+---+-------+-------|r|---+-----------|r|------M--
          |   |       | |   |   |       |       | |   |           | |
    c2: --+---+-------| |---+---+-------+-------| |---+-----------| |---------
          |   |       | |   |   |       |       | |   |           | |
    a2: --+---+---X---| |---+---+---X---+---X---| |---+---X---X---| |---X-----
          |   |   |   | |   |   |   |   |   |   | |   |   |   |   | |   |
    b3: --+---+---+---| |---+---+---+---+---+---| |---+---+---+---| |---+--M--
          |   |   |    -    |   |   |   |   |    -    |   |   |    -    |
    y0: --@---+---+---------@---+---+---@---+---------@---+---@---------@-----
          |   |   |         |   |   |   |   |         |   |   |         |
    y1: --+---@---+---------+---@---+---+---@---------+---@---+---------+-----
          |   |   |         |   |   |   |   |         |   |   |         |
    y2: --+---+---@---------+---+---@---+---+---------+---+---+---------+-----
          |   |   |         |   |   |   |   |         |   |   |         |
    x0: --@---@---@---------@---@---@---+---+---------+---+---+---------+-----
                                        |   |         |   |   |         |
    x1: --------------------------------@---@---------@---@---+---------+-----
                                                              |         |
    x2: ------------------------------------------------------@---------@-----
    """

    def __init__(self, num_qubits):
        super(Multiplier, self)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        n = int(len(qubits)/5)
        # c = qubits[0:n*3:3]
        a = qubits[1:n*3:3]
        # b = qubits[2::3]
        y = qubits[n*3:n*4]
        x = qubits[n*4:]

        for i, x_i in enumerate(x):
            # a = (y*(2**i))*x_i
            for a_qubit, y_qubit in zip(a[i:], y[:n-i]):
                yield cirq.TOFFOLI(x_i, y_qubit, a_qubit)
            # b += a
            yield Adder(3 * n).on(*qubits[:3 * n])
            # a = 0
            for a_qubit, y_qubit in zip(a[i:], y[:n-i]):
                yield cirq.TOFFOLI(x_i, y_qubit, a_qubit)


def init_qubits(x_bin, *qubits):
    for x, qubit in zip(x_bin, list(qubits)[::-1]):
        yield cirq.X(qubit)**float(x)

def init_qubits_sym(x_bin, *qubits):
    for x, qubit in zip(x_bin, list(qubits)[::-1]):
        yield cirq.X(qubit)**x

def experiment_adder(n=3):

    qubits = cirq.LineQubit.range(3 * n)
    # c = dm_qubits[0::3]
    a = qubits[1::3]
    b = qubits[2::3]

    kc_circuit = cirq.Circuit(
        init_qubits_sym([sympy.Symbol('a_bin'+str(index)) for index in range(n)], *a),
        init_qubits_sym([sympy.Symbol('b_bin'+str(index)) for index in range(n)], *b),
        cirq.decompose(Adder(n * 3).on(*qubits)),
        cirq.measure(*b, key='result')
    )
    kc_simulator = cirq.KnowledgeCompilationSimulator(kc_circuit, initial_state=0, intermediate=False)

    for p in range(2**n):
        for q in range(2**n):
            a_bin = '{:08b}'.format(p)[-n:]
            b_bin = '{:08b}'.format(q)[-n:]

            a_bin_dict = { 'a_bin'+str(index):a_bin[index] for index in range(n) }
            b_bin_dict = { 'b_bin'+str(index):b_bin[index] for index in range(n) }
            param_resolver = cirq.ParamResolver({**a_bin_dict,**b_bin_dict})

            dm_circuit = cirq.Circuit(
                init_qubits(a_bin, *a),
                init_qubits(b_bin, *b),
                Adder(n * 3).on(*qubits)
            )
            # np.testing.assert_almost_equal(
            #     cirq.Simulator().simulate(dm_circuit).state_vector(),
            #     kc_simulator.simulate(kc_circuit,param_resolver=param_resolver).state_vector()
            # )

            sv_circuit = cirq.Circuit(
                init_qubits(a_bin, *a),
                init_qubits(b_bin, *b),
                Adder(n * 3).on(*qubits),
                cirq.measure(*b, key='result')
            )
            sv_result = cirq.Simulator().run(sv_circuit, repetitions=1).measurements['result']
            sv_sum_bin = ''.join(sv_result[0][::-1].astype(int).astype(str))

            kc_result = kc_simulator.run(kc_circuit, param_resolver=param_resolver, repetitions=1).measurements['result']
            kc_sum_bin = ''.join(kc_result[0][::-1].astype(int).astype(str))

            print ('{} + {} = {}'.format(a_bin, b_bin, sv_sum_bin))
            print ('{} + {} = {}'.format(a_bin, b_bin, kc_sum_bin))
            assert sv_sum_bin==kc_sum_bin


def experiment_multiplier(n=2):

    qubits = cirq.LineQubit.range(5 * n)
    # c = qubits[0:n*3:3]
    # a = qubits[1:n*3:3]
    b = qubits[2:n*3:3]
    y = qubits[n*3:n*4]
    x = qubits[n*4:]

    kc_circuit = cirq.Circuit(
        init_qubits_sym([sympy.Symbol('x_bin'+str(index)) for index in range(n)], *x),
        init_qubits_sym([sympy.Symbol('y_bin'+str(index)) for index in range(n)], *y),
        cirq.decompose(Multiplier(5 * n).on(*qubits)),
        cirq.measure(*b, key='result')
    )
    cirq.optimizers.SynchronizeTerminalMeasurements().optimize_circuit(kc_circuit)
    kc_simulator = cirq.KnowledgeCompilationSimulator(kc_circuit, initial_state=0, intermediate=False)

    for p in range(2**n):
        for q in range(2**n):
            y_bin = '{:08b}'.format(p)[-n:]
            x_bin = '{:08b}'.format(q)[-n:]

            x_bin_dict = { 'x_bin'+str(index):x_bin[index] for index in range(n) }
            y_bin_dict = { 'y_bin'+str(index):y_bin[index] for index in range(n) }
            param_resolver = cirq.ParamResolver({**x_bin_dict,**y_bin_dict})

            dm_circuit = cirq.Circuit(
                init_qubits(x_bin, *x),
                init_qubits(y_bin, *y),
                Multiplier(5 * n).on(*qubits),
            )
            np.testing.assert_almost_equal(
                cirq.Simulator().simulate(dm_circuit).state_vector(),
                kc_simulator.simulate(kc_circuit,param_resolver=param_resolver).state_vector()
            )

            sv_circuit = cirq.Circuit(
                init_qubits(x_bin, *x),
                init_qubits(y_bin, *y),
                Multiplier(5 * n).on(*qubits),
                cirq.measure(*b, key='result')
            )
            sv_result = cirq.Simulator().run(sv_circuit, repetitions=1).measurements['result']
            sv_sum_bin = ''.join(sv_result[0][::-1].astype(int).astype(str))

            kc_result = kc_simulator.run(kc_circuit, param_resolver=param_resolver, repetitions=1).measurements['result']
            kc_sum_bin = ''.join(kc_result[0][::-1].astype(int).astype(str))

            print ('{} * {} = {}'.format(y_bin, x_bin, sv_sum_bin))
            print ('{} * {} = {}'.format(y_bin, x_bin, kc_sum_bin))
            assert sv_sum_bin==kc_sum_bin

def main():
    print ('Execute Adder')
    experiment_adder(3)
    print ('')
    print ('Execute Multiplier')
    experiment_multiplier(1)

if __name__ == '__main__':
    main()
