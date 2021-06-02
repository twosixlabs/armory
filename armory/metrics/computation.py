"""
Computational metrics
"""

import collections
import contextlib
import io
import logging
import time

import cProfile
import pstats


logger = logging.getLogger(__name__)


class NullProfiler:
    """
    Measures computational resource use
    """

    def __init__(self):
        self.measurement_dict = {}

    @contextlib.contextmanager
    def measure(self, name):
        yield

    def results(self):
        return {}


class BasicProfiler(NullProfiler):
    @contextlib.contextmanager
    def measure(self, name):
        startTime = time.perf_counter()
        yield
        elapsedTime = time.perf_counter() - startTime

        if name not in self.measurement_dict:
            self.measurement_dict[name] = {
                "execution_count": 0,
                "total_time": 0,
            }
        comp = self.measurement_dict[name]
        comp["execution_count"] += 1
        comp["total_time"] += elapsedTime

    def results(self):
        results = {}
        for name, entry in self.measurement_dict.items():
            if "execution_count" not in entry or "total_time" not in entry:
                logger.warning(
                    "Computation resource dictionary entry {name} corrupted, missing data."
                )
                continue
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
        return results


class DeterministicProfiler(NullProfiler):
    def __init__(self):
        super().__init__()
        logger.warning(
            "Using Deterministic profiler. This may reduce timing accuracy and result in a large results file."
        )

    @contextlib.contextmanager
    def measure(self, name):
        pr = cProfile.Profile()
        pr.enable()
        startTime = time.perf_counter()
        yield
        elapsedTime = time.perf_counter() - startTime
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()

        if name not in self.measurement_dict:
            self.measurement_dict[name] = {
                "execution_count": 0,
                "total_time": 0,
                "stats": "",
            }
        comp = self.measurement_dict[name]
        comp["execution_count"] += 1
        comp["total_time"] += elapsedTime
        comp["stats"] += stats

    def results(self):
        results = {}
        for name, entry in self.measurement_dict.items():
            if any(x not in entry for x in ("execution_count", "total_time", "stats")):
                logger.warning(
                    "Computation resource dictionary entry {name} corrupted, missing data."
                )
                continue
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
            results[f"{name} profiler stats"] = entry["stats"]
        return results


_PROFILERS = {
    None: NullProfiler,
    "basic": BasicProfiler,
    "deterministic": DeterministicProfiler,
}


def profiler_class(name=None):
    """
    Return profiler class by name
    """
    if isinstance(name, str):
        name = name.lower()
    elif not name:
        name = None

    try:
        return _PROFILERS[name]
    except KeyError:
        raise KeyError(f"Profiler {name} is not in {tuple(_PROFILERS)}.")


def profiler_from_config(config):
    return profiler_class(config.get("profiler_type"))()


@contextlib.contextmanager
def resource_context(name="Name", profiler=None, computational_resource_dict=None):
    logger.warning("DEPRECATED. Use armory.metrics.computation.Profiler.measure")
    if profiler is None:
        yield
        return 0
    profiler_types = ["Basic", "Deterministic"]
    if profiler is not None and profiler not in profiler_types:
        raise ValueError(f"Profiler {profiler} is not one of {profiler_types}.")
    if profiler == "Deterministic":
        logger.warn(
            "Using Deterministic profiler. This may reduce timing accuracy and result in a large results file."
        )
        pr = cProfile.Profile()
        pr.enable()
    startTime = time.perf_counter()
    yield
    elapsedTime = time.perf_counter() - startTime
    if profiler == "Deterministic":
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()
    if name not in computational_resource_dict:
        computational_resource_dict[name] = collections.defaultdict(lambda: 0)
        if profiler == "Deterministic":
            computational_resource_dict[name]["stats"] = ""
    comp = computational_resource_dict[name]
    comp["execution_count"] += 1
    comp["total_time"] += elapsedTime
    if profiler == "Deterministic":
        comp["stats"] += stats
    return 0
