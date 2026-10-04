"""The shell tool marks its call as failed exactly where it returns a failure as text.

Tau3's shell does not raise for a command that fails; it returns the failure as text, so the tool message's
``error`` stays false. Telemetry records the shell's own decision instead, and parses no text.
"""

import pytest

from tau2.domains.banking_knowledge.retrieval_mixins import ShellMixin
from tau2.utils import telemetry


class StandInSandbox:
    def __init__(self, result):
        self.result = result

    def run_command(self, command):
        return self.result


class Shell(ShellMixin):
    def __init__(self, sandbox):
        self._sandbox = sandbox


@pytest.mark.parametrize(
    "command, result, text, failed",
    [
        ("cat INDEX.md", (0, "index", ""), "index", False),
        ("grep -r x .", (1, "", ""), "No matches found.", False),  # Tau3: not an error
        (
            "cat missing.md",
            (1, "", "no such file"),
            "Error (exit code 1): no such file",
            True,
        ),
        (
            "grep -r x /",
            (1, "", "Error: Command blocked - contains 'absolute path: /'"),
            "Error (exit code 1): Error: Command blocked - contains 'absolute path: /'",
            True,
        ),
        (
            "sleep 60",
            (124, "", "Command timed out after 30 seconds"),
            "Error (exit code 124): Command timed out after 30 seconds",
            True,
        ),
        ("false", (2, "", ""), "Command failed with exit code 2", True),
    ],
)
def test_the_shell_marks_the_failures_it_returns(command, result, text, failed):
    with telemetry.tool_call() as outcome:
        assert Shell(StandInSandbox(result)).shell(command) == text
    assert outcome == {"reported_failure": failed}


def test_without_a_sandbox_the_shell_reports_a_failure():
    with telemetry.tool_call() as outcome:
        assert Shell(None).shell("ls") == "Error: Sandbox not initialized"
    assert outcome == {"reported_failure": True}
