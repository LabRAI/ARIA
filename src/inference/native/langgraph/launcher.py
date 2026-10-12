"""Isolated native StateGraph execution. Called only by the YAML runner."""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

from common.artifacts import atomic_write_json
from common.model_api import ApiConfig, ModelApiClient
from inference.native.model_hooks import intercept_native_text
from inference.native.runtime_hooks import runtime_from_bootstrap


class DebateState(TypedDict):
    messages: list[dict[str, str]]
    rounds: int


def execute(request: dict, *, client=None, aria=None) -> dict:
    from langgraph.graph import END, START, StateGraph

    settings = {f.name: request["api"][f.name] for f in fields(ApiConfig)
                if f.name in request["api"]}
    client = client or ModelApiClient(ApiConfig(**settings))
    aria = aria if aria is not None else runtime_from_bootstrap(request.get("aria"))
    rounds = int(request["rounds"])
    if rounds < 1:
        raise ValueError("LangGraph max_turns (debate rounds) must be positive")
    trace, transcript = [], []
    roles = {
        "solver": "Solve the task. Consider the critic's feedback when revising.",
        "critic": "Independently check the solver's reasoning. Identify concrete errors and corrections.",
        "synthesizer": "Resolve the debate and deliver the final answer. End with FINAL_ANSWER: followed by only the answer value or option label, without Markdown or explanation on that line.",
    }

    def node(name):
        def run(state):
            history = list(state["messages"])
            if history[-1]["role"] == "assistant":
                history.append({"role": "user", "content": f"Continue as {name}. {roles[name]}"})
            messages = [{"role": "system", "content": f"You are the {name}. {roles[name]}"}, *history]
            messages = [{"role": item["role"], "content": item["content"]} for item in messages]
            response = client.invoke(messages)
            delivered = intercept_native_text(aria, response.text, messages, role=name)
            # Substitution occurs BEFORE the message enters graph state/downstream prompts.
            message = {"role": "assistant", "content": delivered, "name": name}
            turn = len(transcript) + 1
            transcript.append({"turn": turn, "role": name, "content": delivered})
            trace.append({"kind": "node_delivery", "turn": turn, "role": name,
                          "original_content": response.text, "content": delivered,
                          "usage": response.usage, "latency_seconds": response.latency_seconds,
                          "response_id": response.response_id})
            return {"messages": [*history, message],
                    "rounds": state["rounds"] + int(name == "critic")}
        return run

    graph = StateGraph(DebateState)
    for name in roles:
        graph.add_node(name, node(name))
    graph.add_edge(START, "solver")
    graph.add_edge("solver", "critic")
    graph.add_conditional_edges("critic", lambda state: "solver" if state["rounds"] < rounds else "synthesizer")
    graph.add_edge("synthesizer", END)
    state = graph.compile().invoke(
        {"messages": [{"role": "user", "content": request["task"]}], "rounds": 0},
        {"recursion_limit": 2 * rounds + 4},
    )
    return {"status": "completed", "error": None, "messages": state["messages"],
            "transcript": transcript, "trace": trace,
            "final_output": state["messages"][-1]["content"],
            "framework_version": version("langgraph"),
            "aria_export": aria.export_state() if aria is not None else None}


if __name__ == "__main__":
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    try:
        output = execute(request)
    except Exception as exc:
        output = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    atomic_write_json(request_path.with_name("langgraph_output.json"), output)
    sys.exit(0 if output["status"] == "completed" else 1)
