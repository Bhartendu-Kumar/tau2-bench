"""Passive run telemetry: attempt, instance and component keys, instance records with discarded trajectories,
tool execution time and tool-reported failures.

The runs below go through Tau3's own runner, retry loop, orchestrator and evaluator on the mock domain. Only the
model is scripted, so no model is called.
"""

import json
from collections import defaultdict
from contextlib import contextmanager

import pytest

import tau2.agent.llm_agent
import tau2.runner.simulation
import tau2.user.user_simulator
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason, TextRunConfig
from tau2.environment.environment import Environment
from tau2.runner import batch, run_tasks
from tau2.runner.progress import run_with_retry
from tau2.user.user_simulator_base import STOP
from tau2.utils import telemetry


class ScriptedModel:
    """Replaces `generate` for the agent and the user simulator, and records the telemetry seen at each call."""

    def __init__(self, fail_instances: int = 0, fail_at: int = 0):
        self.fail_instances = (
            fail_instances  # the first N instances fail on an agent call
        )
        self.fail_at = fail_at  # which of their agent calls fails: 0 is the first
        self.instances: list[str] = []
        self.steps: dict[tuple, int] = defaultdict(int)
        self.calls: list[dict] = []

    def __call__(self, model, messages, tools=None, call_name=None, **kwargs):
        instance = batch._current_simulation_id.get()
        if instance not in self.instances:
            self.instances.append(instance)
        attempt = telemetry.attempt.get()
        self.calls.append(
            {
                "call_name": call_name,
                "instance": instance,
                "component": telemetry.component.get(),
                "attempt": (attempt["task_id"], attempt["trial"]) if attempt else None,
                "keys": telemetry.keys(),
            }
        )
        n = self.steps[instance, call_name]
        self.steps[instance, call_name] += 1
        if call_name == "user_simulator_response":
            return AssistantMessage(
                role="assistant", content="Please list the users." if n == 0 else STOP
            )
        if self.instances.index(instance) < self.fail_instances and n == self.fail_at:
            raise ValueError("unusable agent output")
        if n == 0:
            return AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="call-1", name="get_users", arguments={})],
            )
        return AssistantMessage(role="assistant", content="Here are the users.")


@contextmanager
def scripted(monkeypatch, model):
    monkeypatch.setattr(tau2.agent.llm_agent, "generate", model)
    monkeypatch.setattr(tau2.user.user_simulator, "generate", model)
    seen = {}
    get_response = Environment.get_response
    evaluate = tau2.runner.simulation.evaluate_simulation

    def recording_get_response(self, message):
        seen["tool"] = telemetry.component.get()
        seen["tool_keys"] = telemetry.keys()
        return get_response(self, message)

    def recording_evaluate(*args, **kwargs):
        seen["evaluation"] = telemetry.component.get()
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(Environment, "get_response", recording_get_response)
    monkeypatch.setattr(
        tau2.runner.simulation, "evaluate_simulation", recording_evaluate
    )
    yield seen


def run(domain_name, task, tmp_path, max_retries=3):
    config = TextRunConfig(
        domain=domain_name,
        agent="llm_agent",
        user="user_simulator",
        llm_agent="scripted",
        llm_user="scripted",
        max_retries=max_retries,
        retry_delay=0.0,
    )
    results = run_tasks(
        config, [task], save_path=tmp_path / "results.json", console_display=False
    )
    return results.simulations[0], json.loads((tmp_path / "results.json").read_text())


def test_requests_carry_attempt_instance_component_and_step(
    monkeypatch, domain_name, base_task, tmp_path
):
    model = ScriptedModel()
    with scripted(monkeypatch, model) as seen:
        simulation, _ = run(domain_name, base_task, tmp_path)

    assert simulation.termination_reason == TerminationReason.USER_STOP
    assert [c["call_name"] for c in model.calls] == [
        "user_simulator_response",
        "agent_response",
        "agent_response",
        "user_simulator_response",
    ]
    assert {c["instance"] for c in model.calls} == {simulation.id}
    assert {c["attempt"] for c in model.calls} == {(base_task.id, 0)}
    assert [c["component"] for c in model.calls] == [
        ("user_simulator", 0, None),
        ("agent", 1, None),
        ("agent", 3, None),
        ("user_simulator", 4, None),
    ]
    assert seen["tool"] == ("tool", 2, "call-1")
    assert seen["evaluation"] == ("evaluation", None, None)
    attempt_keys = {"task": base_task.id, "trial": 0, "instance": simulation.id}
    assert [c["keys"] for c in model.calls] == [
        {**attempt_keys, "component": name, "step": step, "tool_call_id": None}
        for name, step in [
            ("user_simulator", 0),
            ("agent", 1),
            ("agent", 3),
            ("user_simulator", 4),
        ]
    ]
    assert seen["tool_keys"] == {
        **attempt_keys,
        "component": "tool",
        "step": 2,
        "tool_call_id": "call-1",
    }
    # Nothing leaks out of the run.
    assert telemetry.attempt.get() is None and telemetry.component.get() is None
    assert batch._current_simulation_id.get() is None


def test_tool_messages_record_execution_time(
    monkeypatch, domain_name, base_task, tmp_path
):
    with scripted(monkeypatch, ScriptedModel()):
        simulation, saved = run(domain_name, base_task, tmp_path)

    tools = [m for m in simulation.messages if m.role == "tool"]
    assert len(tools) == 1 and tools[0].execution_time_seconds >= 0
    assert tools[0].reported_failure is False
    saved_tools = [
        m for m in saved["simulations"][0]["messages"] if m["role"] == "tool"
    ]
    assert saved_tools[0]["execution_time_seconds"] == tools[0].execution_time_seconds


def test_a_tool_that_reports_a_failure_marks_its_message(
    monkeypatch, domain_name, base_task, tmp_path
):
    get_response = Environment.get_response

    def failing(self, message):  # as the shell does for a command that fails
        telemetry.tool_reported_failure()
        return get_response(self, message)

    monkeypatch.setattr(Environment, "get_response", failing)
    with scripted(monkeypatch, ScriptedModel()):
        simulation, saved = run(domain_name, base_task, tmp_path)

    [tool] = [m for m in simulation.messages if m.role == "tool"]
    assert tool.reported_failure is True and tool.error is False
    assert [
        m["reported_failure"]
        for m in saved["simulations"][0]["messages"]
        if m["role"] == "tool"
    ] == [True]


def test_a_tool_outcome_belongs_to_its_call():
    telemetry.tool_reported_failure()  # outside a tool call: nothing to mark
    with telemetry.tool_call() as first:
        telemetry.tool_reported_failure()
    with telemetry.tool_call() as second:
        pass
    assert (first, second) == ({"reported_failure": True}, {"reported_failure": False})
    assert telemetry.tool_outcome.get() is None


def test_kept_instance_record_is_saved(monkeypatch, domain_name, base_task, tmp_path):
    with scripted(monkeypatch, ScriptedModel()):
        simulation, saved = run(domain_name, base_task, tmp_path)

    [record] = saved["simulations"][0]["info"]["instances"]
    assert (
        record["id"] == simulation.id
        and record["status"] == "completed"
        and record["error"] is None
    )
    assert (
        record["component"] == "evaluation"
    )  # the last component the instance entered
    assert record["start"] <= record["end"] and record["thread_cpu_seconds"] >= 0
    if record["on_cpu_seconds"] is not None:  # Linux scheduler statistics
        assert record["on_cpu_seconds"] >= 0 and record["run_queue_wait_seconds"] >= 0


def test_agent_failure_is_saved_with_its_cause_without_another_instance(
    monkeypatch, domain_name, base_task, tmp_path
):
    model = ScriptedModel(fail_instances=2)
    with scripted(monkeypatch, model):
        simulation, saved = run(domain_name, base_task, tmp_path)

    assert simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    records = saved["simulations"][0]["info"]["instances"]
    assert [r["status"] for r in records] == ["ValueError"]
    assert [r["component"] for r in records] == ["agent"]
    assert records[0]["error"] == "unusable agent output"
    assert [r["id"] for r in records] == model.instances
    assert simulation.id not in model.instances  # the failed placeholder never ran
    assert [m["role"] for m in records[0]["trajectory"]] == ["assistant", "user"]


def test_a_discarded_instance_keeps_its_trajectory(
    monkeypatch, domain_name, base_task, tmp_path
):
    model = ScriptedModel(fail_instances=1, fail_at=1)  # after its tool call
    with scripted(monkeypatch, model):
        simulation, saved = run(domain_name, base_task, tmp_path)

    [discarded] = saved["simulations"][0]["info"]["instances"]
    trajectory = discarded["trajectory"]
    assert [m["role"] for m in trajectory] == ["assistant", "user", "assistant", "tool"]
    assert trajectory[2]["tool_calls"][0]["name"] == "get_users"
    assert trajectory[3]["execution_time_seconds"] >= 0
    assert trajectory[3]["reported_failure"] is False
    assert [m["turn_idx"] for m in trajectory] == [0, 1, 2, 3]  # as Tau3 numbers them
    assert simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    assert simulation.info["retry_stopped_by"] == "agent_failure"
    assert simulation.messages == []


def test_a_trajectory_that_cannot_be_read_is_recorded_not_raised():
    def unreadable():
        raise RuntimeError("no trajectory")

    with telemetry.in_attempt("t", 0):
        telemetry.instance_started("i")
        telemetry.instance_trajectory(unreadable)
        telemetry.instance_ended(ValueError("agent failed"))
        [record] = telemetry.attempt.get()["instances"]
    assert record["status"] == "ValueError"
    assert record["trajectory_error"] == "RuntimeError: no trajectory"
    assert "trajectory" not in record and "_trajectory" not in record


def test_known_agent_failure_stops_the_retry_loop(
    monkeypatch, domain_name, base_task, tmp_path
):
    model = ScriptedModel(fail_instances=10)
    with scripted(monkeypatch, model):
        simulation, saved = run(domain_name, base_task, tmp_path, max_retries=2)

    assert simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    records = saved["simulations"][0]["info"]["instances"]
    assert [r["status"] for r in records] == ["ValueError"]
    assert simulation.info["failed_after_attempts"] == 1
    assert simulation.info["retry_stopped_by"] == "agent_failure"
    assert simulation.id not in [r["id"] for r in records]  # the placeholder never ran


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "ValueError", "component": "agent", "error": "candidate failed"}, True),
        ({"status": "RateLimitError", "component": "agent", "error": "429"}, False),
        (
            {
                "status": "ValueError",
                "component": "evaluation",
                "error": "Tool call:\nx\n\nReturned:\ny\n\nExpected:\nz",
            },
            True,
        ),
        ({"status": "ValueError", "component": "evaluation", "error": "grader failed"}, False),
        ({"status": "OSError", "component": "runner", "error": "disk full"}, False),
    ],
)
def test_only_confident_agent_failures_stop_retries(record, expected):
    with telemetry.in_attempt("task", 0):
        telemetry.attempt.get()["instances"].append(record)
        assert telemetry.known_agent_failure() is expected


def test_model_service_failure_keeps_the_retry_loop(base_task):
    RateLimitError = type("RateLimitError", (Exception,), {})
    attempts = []

    def run_fn():
        attempts.append(len(attempts))
        telemetry.instance_started(f"instance-{len(attempts)}")
        error = RateLimitError("rate limited")
        with telemetry.in_component("agent"):
            telemetry.instance_ended(error)
        raise error

    with telemetry.in_attempt(base_task.id, 0):
        simulation = run_with_retry(
            run_fn,
            base_task,
            0,
            1,
            max_retries=2,
            retry_delay=0.0,
            console_display=False,
        )

    assert len(attempts) == 3
    assert simulation.info["failed_after_attempts"] == 3
    assert "retry_stopped_by" not in simulation.info


def test_runner_error_after_a_completed_instance_names_the_runner(base_task):
    attempts = []

    def run_fn():
        attempts.append(len(attempts))
        telemetry.instance_started(f"instance-{len(attempts)}")
        telemetry.instance_ended(None)
        return SimulationRun(
            id=f"instance-{len(attempts)}",
            task_id=base_task.id,
            start_time="s",
            end_time="e",
            duration=0.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
        )

    def save_fn(simulation):
        if len(attempts) == 1:
            raise OSError("disk full")

    with telemetry.in_attempt(base_task.id, 0):
        simulation = run_with_retry(
            run_fn,
            base_task,
            0,
            1,
            retry_delay=0.0,
            console_display=False,
            save_fn=save_fn,
        )

    first, kept = simulation.info["instances"]
    assert (first["status"], first["component"], first["error"]) == (
        "OSError",
        "runner",
        "disk full",
    )
    assert kept["status"] == "completed" and kept["id"] == simulation.id


def test_without_an_attempt_nothing_is_recorded():
    telemetry.instance_started("instance")
    telemetry.instance_ended(None)
    simulation = SimulationRun(
        id="x",
        task_id="t",
        start_time="s",
        end_time="e",
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )
    telemetry.attach_instances(simulation)
    assert simulation.info is None
    assert telemetry.keys() == dict.fromkeys(
        ["task", "trial", "instance", "component", "step", "tool_call_id"]
    )


def test_keys_name_the_current_attempt_instance_and_component():
    with telemetry.in_attempt("task-9", 3):
        telemetry.instance_started("first")
        telemetry.instance_ended(ValueError("discarded"))
        telemetry.instance_started("second")
        with telemetry.in_component("tool", 5, "call-7"):
            assert telemetry.keys() == {
                "task": "task-9",
                "trial": 3,
                "instance": "second",
                "component": "tool",
                "step": 5,
                "tool_call_id": "call-7",
            }


@pytest.mark.parametrize("raised", [ValueError("boom"), KeyboardInterrupt()])
def test_component_is_reset_when_its_step_raises(raised):
    with pytest.raises(type(raised)), telemetry.in_component("agent", 7):
        raise raised
    assert telemetry.component.get() is None
