"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        info = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a coding agent working in a terminal sandbox. "
            "Use the execute tool to inspect the repository and change files. "
            "Each command runs in a new subshell, so set cwd and env instead of "
            "relying on cd or export. Prefer commands that produce little output. "
            "When the task is complete, call send_message with a short summary.\n"
            f"<system_information>\n{info}\n</system_information>"
        )
        self.task_prompt = self.task
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            # Link each observation to the call it answers; IDs may repeat
            # across the trajectory, so match within this action only.
            call_id = call.get("id", "") if isinstance(call, dict) else ""
            function = (call.get("function") or {}) if isinstance(call, dict) else {}
            name = function.get("name")
            raw_arguments = function.get("arguments") or ""

            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except (json.JSONDecodeError, TypeError):
                content = (
                    "Error: tool arguments are not valid JSON. Received "
                    f"{raw_arguments!r}. Resend the call with a valid JSON object."
                )
            else:
                if not isinstance(arguments, dict):
                    content = "Error: tool arguments must be a JSON object."
                elif name == "execute":
                    command = arguments.get("command")
                    if not isinstance(command, (str, list)):
                        content = (
                            "Error: 'execute' requires a 'command' string or argv list."
                        )
                    else:
                        # Optional parameters keep Environment.execute's defaults
                        # (None) unless the agent overrides them in the call.
                        result = self.env.execute(
                            command,
                            shell=arguments.get("shell"),
                            cwd=arguments.get("cwd"),
                            timeout=arguments.get("timeout"),
                            env=arguments.get("env"),
                        )
                        observation = {
                            "output": result.get("output", ""),
                            "returncode": result.get("returncode", -1),
                        }
                        if result.get("exception_info"):
                            observation["exception_info"] = result["exception_info"]
                        content = format_tool_output(observation)
                elif name == "send_message":
                    summary = arguments.get("summary")
                    if not isinstance(summary, str):
                        content = "Error: 'send_message' requires a 'summary' string."
                    else:
                        content = summary
                else:
                    content = (
                        f"Error: unknown tool {name!r}. "
                        "Available tools: execute, send_message."
                    )

            observations.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )
        return observations
