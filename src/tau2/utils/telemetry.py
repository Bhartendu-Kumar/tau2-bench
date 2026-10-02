"""Passive run telemetry: which attempt, instance and component is active, and per-instance thread timing.

An in-process recorder reads these context variables to attribute each model request to its attempt, episode
instance, component (agent, user_simulator, tool, evaluation, setup) and orchestrator step. A request inside an
instance with no component belongs to orchestration. Setting them changes nothing that the agent, the user simulator
or the grader see.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from tau2.utils.utils import get_now

# The attempt being run: {"task_id", "trial", "instances"}, where "instances" lists one record per instance, in order.
attempt: ContextVar[Optional[dict]] = ContextVar("tau2_attempt", default=None)
# The component being run: (name, orchestrator step, tool call id).
component: ContextVar[Optional[tuple]] = ContextVar("tau2_component", default=None)


@contextmanager
def in_attempt(task_id: str, trial: int):
    token = attempt.set({"task_id": task_id, "trial": trial, "instances": []})
    try:
        yield
    finally:
        attempt.reset(token)


@contextmanager
def in_component(
    name: str, step: Optional[int] = None, tool_call_id: Optional[str] = None
):
    token = component.set((name, step, tool_call_id))
    instance = current_instance()
    if instance is not None:
        instance["component"] = (
            name  # the last component entered names the cause if the instance fails
        )
    try:
        yield
    finally:
        component.reset(token)


def current_instance() -> Optional[dict]:
    record = attempt.get()
    return record["instances"][-1] if record and record["instances"] else None


def keys() -> dict:
    """The keys of the work in this context, for a recorder: the attempt (task and trial), the instance, the
    component, the orchestrator step and the tool call. Each is None outside its scope."""
    record = attempt.get()
    instance = current_instance()
    name, step, tool_call_id = component.get() or (None, None, None)
    return {
        "task": record["task_id"] if record else None,
        "trial": record["trial"] if record else None,
        "instance": instance["id"] if instance else None,
        "component": name,
        "step": step,
        "tool_call_id": tool_call_id,
    }


def _thread_times() -> tuple:
    """This thread's CPU time, and its time on a CPU and waiting for one (Linux scheduler statistics, else None)."""
    try:
        with open("/proc/thread-self/schedstat") as f:
            on_cpu, waiting = (int(v) / 1e9 for v in f.read().split()[:2])
    except OSError:
        on_cpu = waiting = None
    return time.thread_time(), on_cpu, waiting


def instance_started(simulation_id: str) -> None:
    record = attempt.get()
    if record is not None:
        record["instances"].append(
            {
                "id": simulation_id,
                "start": get_now(),
                "component": None,
                "_times": _thread_times(),
            }
        )


def instance_ended(exc: Optional[BaseException]) -> None:
    instance = current_instance()
    if instance is None or "_times" not in instance:
        return
    seconds = [
        None if a is None or b is None else b - a
        for a, b in zip(instance.pop("_times"), _thread_times())
    ]
    instance.update(
        end=get_now(),
        status="completed" if exc is None else type(exc).__name__,
        error=None if exc is None else str(exc),
        thread_cpu_seconds=seconds[0],
        on_cpu_seconds=seconds[1],
        run_queue_wait_seconds=seconds[2],
    )


def runner_failed(exc: BaseException) -> None:
    """Mark an instance that completed but was discarded by an error in the runner, such as a failed save."""
    instance = current_instance()
    if instance is not None and instance.get("status") == "completed":
        instance.update(status=type(exc).__name__, error=str(exc), component="runner")


def attach_instances(simulation) -> None:
    """Persist the attempt's instance records with the simulation that represents it."""
    record = attempt.get()
    if record is not None:
        simulation.info = {
            **(simulation.info or {}),
            "instances": list(record["instances"]),
        }
