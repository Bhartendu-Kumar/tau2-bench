"""Model requests carry x-agentrx- headers that attribute them to their run, task, trial, instance, component
and step, so that a model gateway's log can attribute every attempt. Every request goes to a local fake endpoint.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fake_openai import FakeOpenAI

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder
from tau2.knowledge.postprocessors.pointwise_llm_reranker import PointwiseLLMReranker
from tau2.utils import telemetry
from tau2.utils.llm_utils import generate

CALL_ID = re.compile(r"[0-9a-f]{32}")


def agentrx_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: value for name, value in headers.items() if name.startswith("x-agentrx-")
    }


def ask(fake: FakeOpenAI, **kwargs: Any) -> None:
    generate(
        model="openai/gpt-4.1",
        messages=[
            SystemMessage(role="system", content="Be brief."),
            UserMessage(role="user", content="hi"),
        ],
        api_base=fake.base_url,
        api_key="local-placeholder",
        **kwargs,
    )


def test_each_model_call_is_tagged_with_its_work_and_a_new_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(telemetry.RUN_ID_ENV, "run-1")
    with FakeOpenAI() as fake, telemetry.in_attempt("task 7/b", 2):
        telemetry.instance_started("sim-1")
        with telemetry.in_component("agent", step=3, tool_call_id="call_9"):
            ask(fake, call_name="generate_agent_message")
            ask(fake, call_name="generate_agent_message")

    first, second = (agentrx_headers(request.headers) for request in fake.received)
    assert {
        name: value for name, value in first.items() if name != "x-agentrx-call-id"
    } == {
        "x-agentrx-run-id": "run-1",
        "x-agentrx-task": "task%207%2Fb",
        "x-agentrx-trial": "2",
        "x-agentrx-instance": "sim-1",
        "x-agentrx-component": "agent",
        "x-agentrx-step": "3",
        "x-agentrx-tool-call-id": "call_9",
        "x-agentrx-call-name": "generate_agent_message",
    }
    assert CALL_ID.fullmatch(first["x-agentrx-call-id"])
    assert first["x-agentrx-call-id"] != second["x-agentrx-call-id"]


def test_retries_of_a_call_keep_its_call_id_and_unknown_values_are_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(telemetry.RUN_ID_ENV, raising=False)
    with FakeOpenAI() as fake:
        fake.fail_next(429)
        ask(fake, num_retries=1)

    first, retry = (agentrx_headers(request.headers) for request in fake.received)
    assert first == retry
    assert set(first) == {"x-agentrx-call-id"}


def test_embedding_and_reranking_requests_are_tagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with FakeOpenAI() as fake, telemetry.in_attempt("task-1", 0):
        monkeypatch.setenv("OPENAI_BASE_URL", fake.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "local-placeholder")
        with telemetry.in_component("tool", step=4):
            OpenAIEmbedder(model="text-embedding-3-large").embed(["card fees"])
            reranker = PointwiseLLMReranker(model="gpt-4.1", max_concurrency=2)
            reranker.process(
                [("doc-1", 1.0), ("doc-2", 0.5)],
                {"query": "fees"},
                {"doc_content_map": {"doc-1": "Card fees.", "doc-2": "Transfers."}},
            )

    tagged = [
        (request.path, agentrx_headers(request.headers)) for request in fake.received
    ]
    assert [path for path, _ in tagged] == [
        "/v1/embeddings",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    for _, headers in tagged:
        assert (
            headers["x-agentrx-task"],
            headers["x-agentrx-component"],
            headers["x-agentrx-step"],
        ) == ("task-1", "tool", "4")
    assert [headers["x-agentrx-call-name"] for _, headers in tagged] == [
        "embed",
        "rerank",
        "rerank",
    ]
