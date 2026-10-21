"""Passive run telemetry: which attempt, instance and component is active, per-instance thread timing, the
trajectory of a discarded instance, and the failure a tool reports in its result.

An in-process recorder reads these context variables to attribute each model request to its attempt, episode
instance, component (agent, user_simulator, tool, evaluation, setup) and orchestrator step. A request inside an
instance with no component belongs to orchestration. Tau3 drops a discarded instance's trajectory; its record keeps
it. A tool that returns its failure as text, as the shell does, marks its call. Setting them changes nothing that the
agent, the user simulator or the grader see.
"""

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional
from urllib.parse import quote

from tau2.utils.utils import get_now

# The attempt being run: {"task_id", "trial", "instances"}, where "instances" lists one record per instance, in order.
attempt: ContextVar[Optional[dict]] = ContextVar("tau2_attempt", default=None)
# The component being run: (name, orchestrator step, tool call id).
component: ContextVar[Optional[tuple]] = ContextVar("tau2_component", default=None)
# The tool call being made: a tool that returns its failure as text, instead of raising, marks it here.
tool_outcome: ContextVar[Optional[dict]] = ContextVar("tau2_tool_outcome", default=None)

# Model-service and access failures can happen while the agent component is active but are not known to be caused by
# the agent. They remain eligible for Tau3's conversation retry. Other exceptions raised while the agent runs are
# agent failures. Keep this narrow: an uncertain failure should still retry.
RETRYABLE_AGENT_FAILURES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "APIResponseValidationError",
        "APIStatusError",
        "APITimeoutError",
        "AuthenticationError",
        "BadGatewayError",
        "BudgetExceededError",
        "ConflictError",
        "ContentPolicyViolationError",
        "InternalServerError",
        "NotFoundError",
        "PermissionDeniedError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }
)
REPLAY_MISMATCH = ("Tool call:\n", "\n\nReturned:\n", "\n\nExpected:\n")


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


@contextmanager
def tool_call():
    """Collect what the tool called inside reports about its own outcome."""
    outcome = {"reported_failure": False}
    token = tool_outcome.set(outcome)
    try:
        yield outcome
    finally:
        tool_outcome.reset(token)


def tool_reported_failure() -> None:
    """Mark the current tool call as failed, for a tool that returns its failure as text, as the shell does."""
    outcome = tool_outcome.get()
    if outcome is not None:
        outcome["reported_failure"] = True


def instance_trajectory(get_trajectory: Callable[[], list]) -> None:
    """Register how to read the current instance's trajectory, so that a discarded instance keeps it."""
    instance = current_instance()
    if instance is not None:
        instance["_trajectory"] = get_trajectory


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


# The run that a model gateway's log should attribute requests to, set by whoever starts the run.
RUN_ID_ENV = "AGENTRX_RUN_ID"


def request_headers(call_name: Optional[str] = None) -> dict:
    """HTTP headers that tag one model request with the work that made it, so that a model gateway's log can
    attribute the request: the run (from AGENTRX_RUN_ID), the attempt's task and trial, the instance, the component,
    the orchestrator step, the tool call, the call's name, and a new id for this call. A client's retries of the call
    resend the same headers, so the gateway groups them under one call id. Values are percent-encoded so that any
    id is a valid header value; a value that is None is left out."""
    values = {
        "run-id": os.environ.get(RUN_ID_ENV),
        **{name.replace("_", "-"): value for name, value in keys().items()},
        "call-name": call_name,
        "call-id": uuid.uuid4().hex,
    }
    return {
        f"x-agentrx-{name}": quote(str(value), safe="")
        for name, value in values.items()
        if value is not None
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
    get_trajectory = instance.pop("_trajectory", None)
    instance.update(
        end=get_now(),
        status="completed" if exc is None else type(exc).__name__,
        error=None if exc is None else str(exc),
        thread_cpu_seconds=seconds[0],
        on_cpu_seconds=seconds[1],
        run_queue_wait_seconds=seconds[2],
    )
    if exc is not None and get_trajectory is not None:
        # Tau3 drops a discarded instance's trajectory. It is kept here, read after the instance's end, and a
        # failure to read it is recorded, never raised in place of the instance's own exception.
        try:
            instance["trajectory"] = [
                item.model_dump(mode="json") for item in get_trajectory()
            ]
        except Exception as error:
            instance["trajectory_error"] = f"{type(error).__name__}: {error}"


def runner_failed(exc: BaseException) -> None:
    """Mark an instance that completed but was discarded by an error in the runner, such as a failed save."""
    instance = current_instance()
    if instance is not None and instance.get("status") == "completed":
        instance.update(status=type(exc).__name__, error=str(exc), component="runner")


def known_agent_failure() -> bool:
    """Whether the current discarded instance failed for a reason confidently caused by the agent.

    Stop Tau3's conversation retry only for these cases. A failure raised while the agent runs is the agent's unless
    its exception class names a model-service, timeout or access failure. A grader replay mismatch is also the
    agent's: only agent code can change a tool result already recorded in its conversation.
    """
    instance = current_instance() or {}
    status, component_name, error = instance.get("status"), instance.get("component"), instance.get("error") or ""
    if component_name == "agent":
        return status not in RETRYABLE_AGENT_FAILURES
    return (
        component_name == "evaluation"
        and error.startswith(REPLAY_MISMATCH[0])
        and all(part in error for part in REPLAY_MISMATCH[1:])
    )


def attach_instances(simulation) -> None:
    """Persist the attempt's instance records with the simulation that represents it."""
    record = attempt.get()
    if record is not None:
        simulation.info = {
            **(simulation.info or {}),
            "instances": list(record["instances"]),
        }
