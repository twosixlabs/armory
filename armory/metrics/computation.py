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


@contextlib.contextmanager
def resource_context(name="Name", profiler=None, computational_resource_dict=None):
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
