.. image:: _static/Cirq_logo_color.png
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
We are still making breaking changes.
We *will* break your code when we make new releases.
We recommend that you target a specific version of Cirq, and periodically bump to the latest release.
That way you have control over when a breaking change affects you.

User Documentation
------------------

.. toctree::
    :maxdepth: 2

    install
    tutorial
    circuits
    gates
    noise
    simulation
    schedules
    qudits
    development
    examples

Developer Documentation
-----------------------

.. toctree::
    :maxdepth: 1

    dev/index.rst


API Reference
-------------

.. toctree::
    :maxdepth: 2

    api
