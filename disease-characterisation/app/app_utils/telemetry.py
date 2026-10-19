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

"""Observability, OpenTelemetry tracing, structured logging, and PII redaction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
import time
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


class TraceCorrelatedJsonFormatter(logging.Formatter):
    """Formats Python logs as Cloud Logging-compatible JSON correlated with OTel traces."""

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")

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
            payload["exception"] = redact_pii(self.formatException(record.exc_info))

        return json.dumps(payload, default=str)


def get_structured_logger(name: str = "disease_characterisation") -> logging.Logger:
    """Returns a structured logger configured with trace correlation and PII scrubbing."""
    logger = logging.getLogger(name)
    if not any(
        isinstance(h.formatter, TraceCorrelatedJsonFormatter) for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(TraceCorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


telemetry_logger = get_structured_logger("disease_characterisation.telemetry")


class DiseaseObservabilityPlugin(BasePlugin):
    """ADK Plugin for agent/tool latency, LLM token accounting, and PII scrubbing."""

    def __init__(self, name: str = "disease_observability") -> None:
        super().__init__(name=name)
        self._agent_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}

    async def on_user_message_callback(
        self,
        *,
        invocation_context: Any,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Scrubs accidental PII/PHI from user messages before pipeline execution."""
        if not user_message or not user_message.parts:
            return None

        modified = False
        new_parts: list[types.Part] = []
        for part in user_message.parts:
            if getattr(part, "text", None):
                scrubbed = redact_pii(part.text)
                if scrubbed != part.text:
                    modified = True
                    new_parts.append(types.Part.from_text(text=scrubbed))
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

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
        key = f"{id(callback_context)}:{agent.name}"
        self._agent_start_times[key] = time.perf_counter()
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("adk.agent.name", agent.name)
            efo_id = callback_context.state.get("efo_id")
            if efo_id:
                span.set_attribute("disease.efo_id", str(efo_id))
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        key = f"{id(callback_context)}:{agent.name}"
        start = self._agent_start_times.pop(key, None)
        duration_ms = (
            round((time.perf_counter() - start) * 1000.0, 2)
            if start is not None
            else None
        )
        efo_id = callback_context.state.get("efo_id")
        span = trace.get_current_span()
        if span and span.is_recording():
            if duration_ms is not None:
                span.set_attribute("adk.agent.duration_ms", duration_ms)
            if efo_id:
                span.set_attribute("disease.efo_id", str(efo_id))

        telemetry_logger.info(
            f"Agent '{agent.name}' completed in {duration_ms}ms",
            extra={
                "structured_data": {
                    "event_type": "agent_completed",
                    "agent_name": agent.name,
                    "duration_ms": duration_ms,
                    "efo_id": efo_id,
                }
            },
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is None:
            return None

        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidate_tokens = getattr(usage, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        agent_name = getattr(callback_context, "agent_name", "unknown")

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
            span.set_attribute("gen_ai.usage.output_tokens", int(candidate_tokens))
            span.set_attribute("gen_ai.usage.thoughts_tokens", int(thoughts_tokens))
            span.set_attribute("gen_ai.usage.total_tokens", int(total_tokens))

        telemetry_logger.info(
            f"LLM response for '{agent_name}': {total_tokens} total tokens",
            extra={
                "structured_data": {
                    "event_type": "llm_usage",
                    "agent_name": agent_name,
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
        key = f"{id(tool_context)}:{tool.name}"
        self._tool_start_times[key] = time.perf_counter()
        # Scrub any string arguments in-place before external API execution
        for arg_key, arg_val in list(tool_args.items()):
            if isinstance(arg_val, str) and arg_key != "markdown_content":
                tool_args[arg_key] = redact_pii(arg_val)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        key = f"{id(tool_context)}:{tool.name}"
        start = self._tool_start_times.pop(key, None)
        duration_ms = (
            round((time.perf_counter() - start) * 1000.0, 2)
            if start is not None
            else None
        )
        status_val = (
            result.get("status", "unknown")
            if isinstance(result, dict)
            else "unknown"
        )

        span = trace.get_current_span()
        if span and span.is_recording():
            if duration_ms is not None:
                span.set_attribute("adk.tool.duration_ms", duration_ms)
            span.set_attribute("adk.tool.status", str(status_val))
            if status_val == "error":
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        str(result.get("error_message", "Tool returned error status")),
                    )
                )

        log_fn = (
            telemetry_logger.warning
            if status_val in ("error", "not_found")
            else telemetry_logger.info
        )
        log_fn(
            f"Tool '{tool.name}' finished with status='{status_val}' in {duration_ms}ms",
            extra={
                "structured_data": {
                    "event_type": "tool_completed",
                    "tool_name": tool.name,
                    "status": status_val,
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
        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, redact_pii(str(error))))

        telemetry_logger.error(
            f"Tool '{tool.name}' raised exception: {redact_pii(str(error))}",
            extra={
                "structured_data": {
                    "event_type": "tool_exception",
                    "tool_name": tool.name,
                    "error": redact_pii(str(error)),
                }
            },
        )
        return None
