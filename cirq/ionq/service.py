# Copyright 2020 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Service to access IonQs API."""

from typing import Optional, Sequence, TYPE_CHECKING

from cirq import protocols, study

from cirq.ionq import calibration, ionq_client, job, results, serializer

if TYPE_CHECKING:
    import cirq


class Service:
    """A class to access IonQ's API."""

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        default_target: str = None,
        api_version='v0.1',
        max_retry_seconds: int = 3600,
        verbose=False,
    ):
        """Creates the Service to access IonQ's API.

        Args:
            remote_host: The location of the api in the form of an url.
            api_key: A string key which allows access to the api.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the api. Defaults to 'v0.1'.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.
        """
        self._client = ionq_client._IonQClient(
            remote_host=remote_host,
            api_key=api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    def run(
        self,
        circuit: 'cirq.Circuit',
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: study.ParamResolverOrSimilarType = study.ParamResolver({}),
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> study.Result:
        """Run the given circuit on the IonQ API.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        result = self.create_job(resolved_circuit, repetitions, name, target).results()
        if isinstance(result, results.QPUResult):
            return result.to_cirq_result(params=study.ParamResolver(param_resolver))
        else:
            return result.to_cirq_result(params=study.ParamResolver(param_resolver), seed=seed)

    def create_job(
        self,
        circuit: 'cirq.Circuit',
        repetitions: int = 100,
        name: Optional[str] = None,
        target: Optional[str] = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.

        Returns:
            A `cirq.ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        serialized_program = serializer.Serializer().serialize(circuit)
        result = self._client.create_job(
            serialized_program=serialized_program, repetitions=repetitions, target=target, name=name
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result['id'])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on the IonQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `cirq.ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQNotFoundException: If there was no job with the given `job_id`.
            IonQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)

    def list_jobs(
        self, status: Optional[str] = None, limit: int = 100, batch_size: int = 1000
    ) -> Sequence[job.Job]:
        """Lists jobs that have been created on the IonQ API.

        Args:
            status: If supplied will filter to only jobs with this status.
            limit: The maximum number of jobs to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of jobs.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        job_dicts = self._client.list_jobs(status=status, limit=limit, batch_size=batch_size)
        return tuple(job.Job(client=self._client, job_dict=job_dict) for job_dict in job_dicts)

    def get_current_calibration(self) -> calibration.Calibration:
        """Gets the most recent calbration via the API.

        Note that currently there is only one target, so this returns the calibration of that
        target.

        The calibration include device specification (number of qubits, connectivity), as well
        as fidelities and timings of gates.

        Returns:
            A `cirq.ionq.Calibration` containing the device specification and calibrations.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        calibration_dict = self._client.get_current_calibration()
        return calibration.Calibration(calibration_dict=calibration_dict)
