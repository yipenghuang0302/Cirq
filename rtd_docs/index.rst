.. image:: ../docs/images/Cirq_logo_color.png
    :alt: Cirq

Cirq
====

Cirq is a software library for writing, manipulating, and optimizing quantum
circuits and then running them against quantum computers and simulators.
Cirq attempts to expose the details of hardware, instead of abstracting them
away, because, in the Noisy Intermediate-Scale Quantum (NISQ) regime, these
details determine whether or not it is possible to execute a circuit at all.

Alpha Disclaimer
----------------

**Cirq is currently in alpha.**
We may change or remove parts of Cirq's API when making new releases.
To be informed of deprecations and breaking changes, subscribe to the
`cirq-announce google group mailing list <https://groups.google.com/forum/#!forum/cirq-announce>`__.

User Documentation
------------------

.. toctree::
    :maxdepth: 2

    docs/install
    docs/tutorials/basics.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Concepts

    docs/gates.ipynb
    docs/circuits.ipynb
    docs/simulation.ipynb
    docs/noise.ipynb
    docs/devices
    docs/interop.ipynb
    docs/qudits.ipynb
    api


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    docs/tutorials/examples
    docs/tutorials/variational_algorithm.ipynb
    docs/tutorials/QAOA_Demo.ipynb
    docs/tutorials/hidden_linear_function.ipynb
    docs/tutorials/Quantum_Walk.ipynb
    docs/tutorials/Rabi_Demo.ipynb
    docs/tutorials/quantum_chess.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Google Documentation

    docs/google/concepts
    docs/tutorials/google/start
    docs/google/engine
    docs/google/devices
    docs/google/specification
    docs/google/calibration
    docs/google/best_practices


.. toctree::
    :maxdepth: 1
    :caption: Pasqal Documentation

    docs/pasqal/getting_started.ipynb
    docs/pasqal/devices
    docs/pasqal/sampler


.. toctree::
    :maxdepth: 1
    :caption: Developer Documentation

    dev/index.rst
    docs/dev/development
