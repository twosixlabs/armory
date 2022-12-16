"""
Test cases for compute metrics
"""

import pytest

from armory.metrics import compute

pytestmark = pytest.mark.unit


def test_null():
    profiler = compute.NullProfiler()
    with profiler.measure("attack"):
        c = sum(i for i in range(1000))
        assert c == 499500
    assert profiler.results() == {}


def test_basic():
    profiler = compute.BasicProfiler()
    execution_count = 10
    for i in range(10):
        with profiler.measure("fast"):
            num = 1000
            assert sum(i for i in range(num)) == sum(i for i in range(num))

        with profiler.measure("slow"):
            num = 10000
            assert sum(i for i in range(num)) == sum(i for i in range(num))

    results = profiler.results()
    name = "fast"
    fast_key = f"Avg. CPU time (s) for {execution_count} executions of {name}"
    name = "slow"
    slow_key = f"Avg. CPU time (s) for {execution_count} executions of {name}"

    assert fast_key in results
    assert slow_key in results


def test_deterministic(caplog):
    profiler = compute.DeterministicProfiler()
    assert "Using Deterministic profiler" in caplog.text
    execution_count = 10
    for i in range(10):
        with profiler.measure("fast"):
            num = 1000
            assert sum(i for i in range(num)) == sum(i for i in range(num))

        with profiler.measure("slow"):
            num = 10000
            assert sum(i for i in range(num)) == sum(i for i in range(num))

    results = profiler.results()
    name = "fast"
    fast_key = f"Avg. CPU time (s) for {execution_count} executions of {name}"
    name = "slow"
    slow_key = f"Avg. CPU time (s) for {execution_count} executions of {name}"

    assert fast_key in results
    assert slow_key in results
    assert results[slow_key] > results[fast_key]

    name = "fast"
    fast_key = f"{name} profiler stats"
    assert fast_key in results
    name = "slow"
    slow_key = f"{name} profiler stats"
    assert slow_key in results
    for k in fast_key, slow_key:
        value = results[k]
        assert "Ordered by: cumulative time" in value
        assert "ncalls  tottime  percall  cumtime  percall" in value
        assert value.count("\n") > 5


def test_profiler_class_and_profiler_from_config():
    metrics_config = {
        "means": True,
        "perturbation": "linf",
        "record_metric_per_sample": False,
        "task": ["categorical_accuracy"],
    }

    assert compute.profiler_class() is compute.NullProfiler
    assert isinstance(
        compute.profiler_from_config(metrics_config), compute.NullProfiler
    )

    for name, Profiler in [
        (None, compute.NullProfiler),
        ("basic", compute.BasicProfiler),
        ("Basic", compute.BasicProfiler),
        ("deterministic", compute.DeterministicProfiler),
        ("Deterministic", compute.DeterministicProfiler),
    ]:
        assert compute.profiler_class(name=name) is Profiler
        metrics_config["profiler_type"] = name
        assert isinstance(compute.profiler_from_config(metrics_config), Profiler)

    name = "not a profiler name"
    with pytest.raises(KeyError):
        compute.profiler_class(name)
    with pytest.raises(KeyError):
        metrics_config["profiler_type"] = name
        compute.profiler_from_config(metrics_config)
