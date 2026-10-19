# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic trajectory and tool-execution evaluator for ADK traces."""

from __future__ import annotations

from typing import Any

REQUIRED_TOOLS = (
    "resolve_disease_ontology",
    "fetch_known_targets",
    "fetch_treatments_and_trials",
    "fetch_recent_publications",
    "save_dossier_artifact",
)


def evaluate(instance: dict[str, Any]) -> dict[str, Any]:
    """Verifies that all 5 core biomedical tools ran without error in the trace."""
    agent_data = instance.get("agent_data") or {}
    turns = agent_data.get("turns") or []

    called_tools: list[str] = []
    error_tools: list[str] = []
    active_agents: list[str] = []

    for turn in turns:
        for event in turn.get("events") or []:
            author = event.get("author")
            if author and author not in active_agents:
                active_agents.append(author)

            content = event.get("content") or {}
            for part in content.get("parts") or []:
                fn_call = part.get("function_call")
                if isinstance(fn_call, dict) and fn_call.get("name"):
                    name = fn_call["name"]
                    if name != "transfer_to_agent":
                        called_tools.append(name)

                fn_resp = part.get("function_response")
                if isinstance(fn_resp, dict) and fn_resp.get("name"):
                    r_name = fn_resp["name"]
                    resp_payload = fn_resp.get("response") or {}
                    if (
                        isinstance(resp_payload, dict)
                        and resp_payload.get("status") == "error"
                    ):
                        error_tools.append(r_name)

    called_set = set(called_tools)
    matched = [t for t in REQUIRED_TOOLS if t in called_set]
    missing = [t for t in REQUIRED_TOOLS if t not in called_set]

    coverage_score = len(matched) / len(REQUIRED_TOOLS)
    penalty = 0.2 * len(set(error_tools))
    final_score = round(max(0.0, coverage_score - penalty), 4)

    explanation = (
        f"Matched {len(matched)}/{len(REQUIRED_TOOLS)} required tools "
        f"({', '.join(matched) if matched else 'none'}). "
        f"Missing: {', '.join(missing) if missing else 'none'}. "
        f"Tool errors: {', '.join(error_tools) if error_tools else 'none'}. "
        f"Sub-agents active: {', '.join(active_agents)}."
    )
    return {
        "score": final_score,
        "explanation": explanation,
    }
