import cirq
import kc_examples.kc_bell_inequality
import kc_examples.kc_bernstein_vazirani
import kc_examples.kc_grover
# import examples.place_on_bristlecone
# import kc_examples.kc_hello_qubit
import kc_examples.kc_quantum_fourier_transform
# import examples.bcs_mean_field
import kc_examples.kc_phase_estimator
import kc_examples.kc_basic_arithmetic
import kc_examples.kc_quantum_teleportation
import kc_examples.kc_superdense_coding

# Standard test runs do not include performance benchmarks.
# coverage: ignore


def test_example_runs_bernstein_vazirani_perf(benchmark):
    benchmark(kc_examples.kc_bernstein_vazirani.main, qubit_count=3)

    # Check empty oracle case. Cover both biases.
    a = cirq.NamedQubit('a')
    assert list(kc_examples.kc_bernstein_vazirani.make_oracle([], a, [], False)) == []
    assert list(kc_examples.kc_bernstein_vazirani.make_oracle([], a, [], True)) == [cirq.X(a)]


# def test_example_runs_hello_line_perf(benchmark):
#     benchmark(examples.place_on_bristlecone.main)


# def test_example_runs_hello_qubit_perf(benchmark):
#     benchmark(examples.hello_qubit.main)


def test_example_runs_bell_inequality_perf(benchmark):
    benchmark(kc_examples.kc_bell_inequality.main)


def test_example_runs_quantum_fourier_transform_perf(benchmark):
    benchmark(kc_examples.kc_quantum_fourier_transform.main)


# def test_example_runs_bcs_mean_field_perf(benchmark):
#     benchmark(examples.bcs_mean_field.main)


def test_example_runs_grover_perf(benchmark):
    benchmark(kc_examples.kc_grover.main)


# def test_example_runs_phase_estimator_perf(benchmark):
#     benchmark(kc_examples.kc_phase_estimator.main, qnums=(2,), repetitions=2)


def test_example_runs_quantum_teleportation(benchmark):
    benchmark(kc_examples.kc_quantum_teleportation.main)


def test_example_runs_superdense_coding(benchmark):
    benchmark(kc_examples.kc_superdense_coding.main)
