# Copyright 2018 The Cirq Developers
#
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

"""Abstract base class for things sampling quantum circuits."""

from typing import Awaitable, List, Union, TYPE_CHECKING, Type

import abc
import asyncio
import collections
import threading

from cirq import circuits, schedules, study, devices
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import cirq


def _int_dtype_of_width(n: int) -> Type[np.number]:
    if n <= 8:
        return np.uint8
    if n <= 16:
        return np.uint16
    if n <= 32:
        return np.uint32
    if n <= 64:
        return np.uint64
    return np.object


def _rep_bits_into_int(reps_then_bits: np.ndarray, *, out: np.ndarray):
    for row in reps_then_bits.transpose():
        out <<= 1
        out += row


def _trial_results_to_dataframe(
        results: List[study.TrialResult]) -> pd.DataFrame:
    # Do some initial scouting of what we're working with.
    samples = 0
    param_keys = set()
    measure_key_widths = collections.defaultdict(lambda: 0)
    for result in results:
        param_keys |= result.params.param_dict.keys()
        samples += result.repetitions
        for k, v in result.measurements.items():
            measure_key_widths[k] = max(measure_key_widths[k], v.shape[1])
    if not param_keys.isdisjoint(measure_key_widths.keys()):
        raise ValueError(
            'Parameter keys overlap with measurement keys.'
            '\nOverlapping keys: {}'.format(
                sorted(param_keys & measure_key_widths.keys())))

    # Prepare buffers to store result data in.
    data = {}
    for k in param_keys:
        data[k] = np.empty(samples, dtype=np.float32)
        data[k][:] = np.NaN
    for k, v in measure_key_widths.items():
        data[k] = np.zeros(samples, dtype=_int_dtype_of_width(v))

    # Write result data into the prepared buffers.
    start = 0
    for result in results:
        end = start + result.repetitions
        for k, v in result.measurements.items():
            _rep_bits_into_int(v, out=data[k][start:end])
        for k, v in result.params.param_dict.items():
            data[k][start:end] = v
        start = end

    return pd.DataFrame(data=data)


class Sampler(metaclass=abc.ABCMeta):
    """Something capable of sampling quantum circuits. Simulator or hardware."""

    def sample(self,
               program: Union[circuits.Circuit, schedules.Schedule],
               repetitions: int = 1,
               *,
               params: 'cirq.Sweepable' = None,
               ) -> pd.DataFrame:
        if params is None:
            params = [study.UnitSweep]
        results = self.run_sweep(program,
                                 params=params,
                                 repetitions=repetitions)
        return _trial_results_to_dataframe(results)

    def run(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> study.TrialResult:
        """Samples from the given Circuit or Schedule.

        By default, the `run_async` method invokes this method on another
        thread. So this method is supposed to be thread safe.

        Args:
            program: The circuit or schedule to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program, study.ParamResolver(param_resolver),
                              repetitions)[0]

    @abc.abstractmethod
    def run_sweep(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            params: study.Sweepable,
            repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Samples from the given Circuit or Schedule.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit or schedule to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """

    async def run_async(self,
                        program: Union[circuits.Circuit, schedules.Schedule], *,
                        repetitions: int) -> Awaitable[study.TrialResult]:
        """Asynchronously samples from the given Circuit or Schedule.

        By default, this method calls `run` on another thread and yields the
        result via the asyncio event loop. However, child classes are free to
        override it to use other strategies.

        Args:
            program: The circuit or schedule to sample from.
            repetitions: The number of times to sample.

        Returns:
            An awaitable TrialResult.
        """
        loop = asyncio.get_event_loop()
        done = loop.create_future()  # type: asyncio.Future

        def run():
            try:
                result = self.run(program, repetitions=repetitions)
            except Exception as exc:
                loop.call_soon_threadsafe(done.set_exception, exc)
            else:
                loop.call_soon_threadsafe(done.set_result, result)

        t = threading.Thread(target=run)
        t.start()
        return await done
