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

"""Observability, OpenTelemetry tracing, intent-vs-outcome structured logging, and PII redaction."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Deterministic PII / PHI regex patterns (carefully scoped so biomedical IDs
# like EFO_0000768, ENSG00000140694, CHEMBL1256391, NCT05571059, PMID, and DOIs
# are never false-positived).
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)
_MRN_RE = re.compile(
    r"\b(?:MRN|Medical\s+Record(?:\s+Number)?|Patient\s+ID)\s*[:#]?\s*[A-Za-z0-9-]{4,}\b",
    re.IGNORECASE,
)
_DOB_RE = re.compile(
    r"\b(?:DOB|Date\s+of\s+Birth)\s*[:#]?\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
    re.IGNORECASE,
)


def redact_pii(text: str) -> str:
    """Scrubs common PII/PHI patterns from text before logging, tracing, or API calls."""
    if not text or not isinstance(text, str):
        return text
    scrubbed = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    scrubbed = _SSN_RE.sub("[REDACTED_SSN]", scrubbed)
    scrubbed = _MRN_RE.sub("[REDACTED_MRN]", scrubbed)
    scrubbed = _DOB_RE.sub("[REDACTED_DOB]", scrubbed)
    scrubbed = _PHONE_RE.sub("[REDACTED_PHONE]", scrubbed)
    return scrubbed


def _sanitize_args_for_log(tool_args: Mapping[str, Any]) -> dict[str, Any]:
    """Produces a PII-scrubbed, compact representation of tool arguments for intent logs."""
    sanitized: dict[str, Any] = {}
    for k, v in tool_args.items():
        if k == "markdown_content" and isinstance(v, str):
            sanitized[k] = f"<markdown:{len(v)} chars>"
        elif isinstance(v, str):
            sanitized[k] = redact_pii(v)[:300]
        elif isinstance(v, (int, float, bool)) or v is None:
            sanitized[k] = v
        else:
            sanitized[k] = redact_pii(str(v))[:200]
    return sanitized


def _describe_tool_intent(
    tool_name: str, sanitized_args: Mapping[str, Any]
) -> str:
    """Builds a human-readable description of the intended action prior to tool execution."""
    if tool_name == "resolve_disease_ontology":
        return (
            f"Normalize disease query '{sanitized_args.get('disease_query', '')}' "
            "to canonical EFO/MONDO and ICD-10 identifiers via Open Targets / OLS4"
        )
    if tool_name == "fetch_known_targets":
        return (
            f"Retrieve top {sanitized_args.get('limit', 10)} associated biological targets "
            f"and tractability profiles for '{sanitized_args.get('efo_id', '')}'"
        )
    if tool_name == "fetch_treatments_and_trials":
        return (
            f"Retrieve approved/investigational drugs and active clinical trials for "
            f"'{sanitized_args.get('disease_name', '')}' ({sanitized_args.get('efo_id', '')})"
        )
    if tool_name == "fetch_recent_publications":
        return (
            f"Retrieve up to {sanitized_args.get('max_results', 5)} peer-reviewed articles "
            f"and bioRxiv/medRxiv preprints for '{sanitized_args.get('disease_name', '')}'"
        )
    if tool_name == "save_dossier_artifact":
        return (
            f"Persist synthesized Markdown dossier artifact and schedule async memory "
            f"consolidation for '{sanitized_args.get('efo_id', '')}'"
        )
    if tool_name == "search_long_term_disease_memory":
        return (
            f"Search persistent long-term vector memory for prior disease knowledge "
            f"matching '{sanitized_args.get('query', '')}'"
        )
    return f"Invoke tool '{tool_name}' with args={json.dumps(dict(sanitized_args), default=str)}"


def _summarize_tool_outcome(result: Any) -> tuple[str, bool, str]:
    """Extracts `(status, intent_fulfilled, outcome_summary)` from a tool response."""
    if not isinstance(result, Mapping):
        summary = redact_pii(str(result))[:240]
        return "completed", True, summary or "Completed non-dict tool execution"

    status_val = str(result.get("status", "success"))
    fulfilled = status_val in ("success", "saved_in_session_state")

    details: list[str] = [f"status={status_val}"]
    if result.get("efo_id"):
        details.append(f"efo_id={result['efo_id']}")
    if "targets" in result and isinstance(result["targets"], list):
        details.append(f"targets_returned={len(result['targets'])}")
    if "known_drugs" in result and isinstance(result["known_drugs"], list):
        details.append(f"drugs_returned={len(result['known_drugs'])}")
    if "clinical_trials" in result and isinstance(
        result["clinical_trials"], list
    ):
        details.append(f"trials_returned={len(result['clinical_trials'])}")
    if "returned_count" in result:
        details.append(f"returned_count={result['returned_count']}")
    if result.get("artifact_filename"):
        details.append(f"artifact={result['artifact_filename']}")
    if result.get("message"):
        details.append(f"msg={redact_pii(str(result['message']))[:140]}")

    return status_val, fulfilled, " | ".join(details)


class TraceCorrelatedJsonFormatter(logging.Formatter):
    """Formats Python logs as Cloud Logging-compatible JSON correlated with OTel traces."""

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self._project_id = project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )

    def format(self, record: logging.LogRecord) -> str:
        message = redact_pii(record.getMessage())
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": message,
        }

        span = trace.get_current_span()
        if span is not None:
            span_ctx = span.get_span_context()
            if span_ctx and span_ctx.is_valid:
                trace_id_hex = format(span_ctx.trace_id, "032x")
                span_id_hex = format(span_ctx.span_id, "016x")
                project_id = self._project_id or os.environ.get(
                    "GOOGLE_CLOUD_PROJECT", ""
                )
                if project_id:
                    payload["logging.googleapis.com/trace"] = (
                        f"projects/{project_id}/traces/{trace_id_hex}"
                    )
                else:
                    payload["trace_id"] = trace_id_hex
                payload["logging.googleapis.com/spanId"] = span_id_hex
                payload["logging.googleapis.com/trace_sampled"] = bool(
                    span_ctx.trace_flags.sampled
                )

        extra_fields = getattr(record, "structured_data", None)
        if isinstance(extra_fields, dict):
            for k, v in extra_fields.items():
                payload[k] = redact_pii(v) if isinstance(v, str) else v

        if record.exc_info:
            payload["exception"] = redact_pii(
                self.formatException(record.exc_info)
            )

        return json.dumps(payload, default=str)


def get_structured_logger(
    name: str = "disease_characterisation",
) -> logging.Logger:
    """Returns a structured logger configured with trace correlation and PII scrubbing."""
    logger = logging.getLogger(name)
    if not any(
        isinstance(h.formatter, TraceCorrelatedJsonFormatter)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(TraceCorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


telemetry_logger = get_structured_logger("disease_characterisation.telemetry")


class DiseaseObservabilityPlugin(BasePlugin):
    """ADK Plugin capturing pre-execution Intent vs. post-execution Outcome, latency, tokens, and PII scrubbing."""

    def __init__(self, name: str = "disease_observability") -> None:
        super().__init__(name=name)
        self._agent_intents: dict[str, dict[str, Any]] = {}
        self._tool_intents: dict[str, dict[str, Any]] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: Any,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Logs incoming user intent (PII-scrubbed) and sanitizes accidental PII/PHI before execution."""
        if not user_message or not user_message.parts:
            return None

        modified = False
        new_parts: list[types.Part] = []
        scrubbed_texts: list[str] = []
        for part in user_message.parts:
            if getattr(part, "text", None):
                scrubbed = redact_pii(part.text)
                scrubbed_texts.append(scrubbed)
                if scrubbed != part.text:
                    modified = True
                    new_parts.append(types.Part.from_text(text=scrubbed))
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        user_preview = " ".join(scrubbed_texts).strip()[:250]
        if user_preview:
            telemetry_logger.info(
                f"User intent received: {user_preview}",
                extra={
                    "structured_data": {
                        "event_type": "user_intent_received",
                        "phase": "intent",
                        "intended_action": f"Fulfill user scoping request: {user_preview}",
                        "pii_redacted": modified,
                    }
                },
            )

        if modified:
            telemetry_logger.warning(
                "Redacted potential PII/PHI from incoming user message.",
                extra={"structured_data": {"event_type": "pii_redacted"}},
            )
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute("security.pii_redacted", True)
            return types.Content(role=user_message.role, parts=new_parts)
        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        """Logs explicit pre-execution agent intent before the agent runs."""
        key = f"{id(callback_context)}:{agent.name}"
        execution_id = f"agent-{uuid.uuid4().hex[:10]}"
        state = getattr(callback_context, "state", {}) or {}
        efo_id = state.get("efo_id")
        role_desc = getattr(agent, "description", "") or f"Execute agent '{agent.name}'"
        intended_action = (
            f"{role_desc} (context efo_id={efo_id})"
            if efo_id and efo_id != "Not yet resolved."
            else role_desc
        )

        self._agent_intents[key] = {
            "start_time": time.perf_counter(),
            "execution_id": execution_id,
            "intended_action": intended_action,
        }

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("adk.agent.name", agent.name)
            span.set_attribute("adk.agent.intent", intended_action)
            span.add_event(
                "agent.intent",
                {
                    "execution_id": execution_id,
                    "agent_name": agent.name,
                    "intended_action": intended_action,
                },
            )
            if efo_id:
                span.set_attribute("disease.efo_id", str(efo_id))

        telemetry_logger.info(
            f"Agent '{agent.name}' starting intended action: {intended_action}",
            extra={
                "structured_data": {
                    "event_type": "agent_intent",
                    "phase": "intent",
                    "execution_id": execution_id,
                    "agent_name": agent.name,
                    "intended_action": intended_action,
                    "efo_id": efo_id,
                }
            },
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        """Logs post-execution agent outcome correlated against pre-execution `intended_action`."""
        key = f"{id(callback_context)}:{agent.name}"
        intent_info = self._agent_intents.pop(key, {})
        start = intent_info.get("start_time")
        execution_id = intent_info.get("execution_id")
        intended_action = (
            intent_info.get("intended_action")
            or getattr(agent, "description", "")
            or f"Execute agent '{agent.name}'"
        )
        duration_ms = (
            round((time.perf_counter() - start) * 1000.0, 2)
            if start is not None
            else None
        )
        state = getattr(callback_context, "state", {}) or {}
        efo_id = state.get("efo_id")
        res_status = str(state.get("resolution_status") or "completed")
        intent_fulfilled = res_status not in ("error", "validation_error")

        span = trace.get_current_span()
        if span and span.is_recording():
            if duration_ms is not None:
                span.set_attribute("adk.agent.duration_ms", duration_ms)
            span.set_attribute("adk.agent.intent_fulfilled", intent_fulfilled)
            span.add_event(
                "agent.outcome",
                {
                    "execution_id": execution_id or "",
                    "agent_name": agent.name,
                    "intent_fulfilled": intent_fulfilled,
                    "resolution_status": res_status,
                },
            )
            if efo_id:
                span.set_attribute("disease.efo_id", str(efo_id))

        telemetry_logger.info(
            f"Agent '{agent.name}' completed in {duration_ms}ms (intent_fulfilled={intent_fulfilled})",
            extra={
                "structured_data": {
                    "event_type": "agent_completed",
                    "phase": "outcome",
                    "execution_id": execution_id,
                    "agent_name": agent.name,
                    "intended_action": intended_action,
                    "outcome_status": res_status,
                    "intent_fulfilled": intent_fulfilled,
                    "duration_ms": duration_ms,
                    "efo_id": efo_id,
                }
            },
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Logs explicit pre-inference model intent (target model, agent, and available tools)."""
        agent_name = getattr(callback_context, "agent_name", "unknown")
        model_name = getattr(llm_request, "model", None) or "default"
        available_tools = sorted(list((llm_request.tools_dict or {}).keys()))
        intended_action = (
            f"Generate next reasoning/tool step for agent '{agent_name}' using model '{model_name}'"
        )

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.request.intended_model", str(model_name))
            span.add_event(
                "llm.intent",
                {
                    "agent_name": str(agent_name),
                    "model": str(model_name),
                    "available_tools_count": len(available_tools),
                },
            )

        telemetry_logger.info(
            f"LLM call intent for '{agent_name}' on model '{model_name}'",
            extra={
                "structured_data": {
                    "event_type": "llm_intent",
                    "phase": "intent",
                    "agent_name": agent_name,
                    "intended_model": model_name,
                    "intended_action": intended_action,
                    "available_tools": available_tools,
                }
            },
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Logs post-inference model outcome, planned tool-call intents, and token accounting."""
        agent_name = getattr(callback_context, "agent_name", "unknown")

        planned_tool_calls: list[dict[str, Any]] = []
        content = getattr(llm_response, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call and getattr(fn_call, "name", None):
                    raw_args = getattr(fn_call, "args", None) or {}
                    safe_args = (
                        _sanitize_args_for_log(raw_args)
                        if isinstance(raw_args, Mapping)
                        else {}
                    )
                    planned_tool_calls.append(
                        {
                            "tool_name": fn_call.name,
                            "intended_args": safe_args,
                        }
                    )

        usage = getattr(llm_response, "usage_metadata", None)
        if usage is None and not planned_tool_calls:
            return None

        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidate_tokens = getattr(usage, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
            span.set_attribute(
                "gen_ai.usage.output_tokens", int(candidate_tokens)
            )
            span.set_attribute(
                "gen_ai.usage.thoughts_tokens", int(thoughts_tokens)
            )
            span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))
            if planned_tool_calls:
                span.set_attribute(
                    "gen_ai.response.planned_tools",
                    ",".join(c["tool_name"] for c in planned_tool_calls),
                )

        telemetry_logger.info(
            f"LLM outcome for '{agent_name}': {total_tokens} total tokens, planned_tools={[c['tool_name'] for c in planned_tool_calls]}",
            extra={
                "structured_data": {
                    "event_type": "llm_usage",
                    "phase": "outcome",
                    "agent_name": agent_name,
                    "planned_tool_calls": planned_tool_calls,
                    "prompt_tokens": prompt_tokens,
                    "candidate_tokens": candidate_tokens,
                    "thoughts_tokens": thoughts_tokens,
                    "total_tokens": total_tokens,
                }
            },
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> Optional[dict[str, Any]]:
        """Scrubs PII in-place and explicitly logs pre-execution `tool_intent` before external calls."""
        for arg_key, arg_val in list(tool_args.items()):
            if isinstance(arg_val, str) and arg_key != "markdown_content":
                tool_args[arg_key] = redact_pii(arg_val)

        call_id = (
            getattr(tool_context, "function_call_id", None)
            or f"call-{uuid.uuid4().hex[:10]}"
        )
        key = f"{id(tool_context)}:{tool.name}:{call_id}"
        sanitized_args = _sanitize_args_for_log(tool_args)
        intended_action = _describe_tool_intent(tool.name, sanitized_args)

        self._tool_intents[key] = {
            "start_time": time.perf_counter(),
            "call_id": call_id,
            "intended_action": intended_action,
            "sanitized_args": sanitized_args,
        }

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("adk.tool.intent", intended_action)
            span.set_attribute(
                "adk.tool.sanitized_args",
                json.dumps(sanitized_args, default=str),
            )
            span.add_event(
                "tool.intent",
                {
                    "call_id": str(call_id),
                    "tool_name": tool.name,
                    "intended_action": intended_action,
                },
            )

        telemetry_logger.info(
            f"Tool '{tool.name}' intent: {intended_action}",
            extra={
                "structured_data": {
                    "event_type": "tool_intent",
                    "phase": "intent",
                    "call_id": call_id,
                    "tool_name": tool.name,
                    "intended_action": intended_action,
                    "sanitized_args": sanitized_args,
                }
            },
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Logs post-execution `tool_completed` outcome paired with pre-execution `intended_action`."""
        call_id = getattr(tool_context, "function_call_id", None)
        key = (
            f"{id(tool_context)}:{tool.name}:{call_id}"
            if call_id
            else next(
                (
                    k
                    for k in reversed(list(self._tool_intents.keys()))
                    if k.startswith(f"{id(tool_context)}:{tool.name}:")
                ),
                "",
            )
        )
        intent_info = self._tool_intents.pop(key, {})
        start = intent_info.get("start_time")
        resolved_call_id = intent_info.get("call_id") or call_id
        sanitized_args = intent_info.get(
            "sanitized_args"
        ) or _sanitize_args_for_log(tool_args)
        intended_action = intent_info.get(
            "intended_action"
        ) or _describe_tool_intent(tool.name, sanitized_args)

        duration_ms = (
            round((time.perf_counter() - start) * 1000.0, 2)
            if start is not None
            else None
        )
        status_val, intent_fulfilled, outcome_summary = _summarize_tool_outcome(
            result
        )

        span = trace.get_current_span()
        if span and span.is_recording():
            if duration_ms is not None:
                span.set_attribute("adk.tool.duration_ms", duration_ms)
            span.set_attribute("adk.tool.status", str(status_val))
            span.set_attribute("adk.tool.intent_fulfilled", intent_fulfilled)
            span.set_attribute("adk.tool.outcome_summary", outcome_summary)
            span.add_event(
                "tool.outcome",
                {
                    "call_id": str(resolved_call_id or ""),
                    "tool_name": tool.name,
                    "status": str(status_val),
                    "intent_fulfilled": intent_fulfilled,
                    "outcome_summary": outcome_summary,
                },
            )
            if status_val == "error":
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        str(
                            result.get(
                                "message",
                                result.get(
                                    "error_message", "Tool returned error status"
                                ),
                            )
                        )
                        if isinstance(result, dict)
                        else "Tool returned error status",
                    )
                )

        log_fn = (
            telemetry_logger.warning
            if not intent_fulfilled
            else telemetry_logger.info
        )
        log_fn(
            f"Tool '{tool.name}' outcome: {outcome_summary} in {duration_ms}ms (intent_fulfilled={intent_fulfilled})",
            extra={
                "structured_data": {
                    "event_type": "tool_completed",
                    "phase": "outcome",
                    "call_id": resolved_call_id,
                    "tool_name": tool.name,
                    "intended_action": intended_action,
                    "sanitized_args": sanitized_args,
                    "status": status_val,
                    "intent_fulfilled": intent_fulfilled,
                    "outcome_summary": outcome_summary,
                    "duration_ms": duration_ms,
                }
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> Optional[dict[str, Any]]:
        """Logs uncaught tool exceptions correlated against the pre-execution `intended_action`."""
        call_id = getattr(tool_context, "function_call_id", None)
        key = f"{id(tool_context)}:{tool.name}:{call_id}" if call_id else ""
        intent_info = self._tool_intents.pop(key, {})
        sanitized_args = intent_info.get(
            "sanitized_args"
        ) or _sanitize_args_for_log(tool_args)
        intended_action = intent_info.get(
            "intended_action"
        ) or _describe_tool_intent(tool.name, sanitized_args)
        scrubbed_err = redact_pii(str(error))

        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(error)
            span.set_attribute("adk.tool.intent_fulfilled", False)
            span.set_status(Status(StatusCode.ERROR, scrubbed_err))

        telemetry_logger.error(
            f"Tool '{tool.name}' raised exception during '{intended_action}': {scrubbed_err}",
            extra={
                "structured_data": {
                    "event_type": "tool_exception",
                    "phase": "outcome",
                    "call_id": intent_info.get("call_id") or call_id,
                    "tool_name": tool.name,
                    "intended_action": intended_action,
                    "sanitized_args": sanitized_args,
                    "intent_fulfilled": False,
                    "error": scrubbed_err,
                }
            },
        )
        return None
