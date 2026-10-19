# ruff: noqa
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

import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools import load_memory, preload_memory
from google.adk.workflow import JoinNode, Workflow
from google.genai import types
from opentelemetry import trace

load_dotenv()

from app.app_utils.telemetry import (
    DiseaseObservabilityPlugin,
    get_structured_logger,
    redact_pii,
)
from app.memory import (
    get_optional_vertex_ai_search_tools,
    search_long_term_disease_memory,
    trigger_async_memory_consolidation,
)
from app.tools import (
    fetch_known_targets,
    fetch_recent_publications,
    fetch_treatments_and_trials,
    resolve_disease_ontology,
    save_dossier_artifact,
)

logger = get_structured_logger("disease_characterisation.routing")
tracer = trace.get_tracer("disease_characterisation.routing")

# Strategic Multi-Tier Model Configuration:
# - FAST_MODEL ("gemini-3.8-flash"): Low-latency deterministic API tool calling & structured field extraction.
# - PRO_MODEL ("gemini-3.8-pro"): Deep cross-domain biomedical synthesis & comparative multi-disease reasoning.
FAST_MODEL = os.environ.get("FAST_MODEL", "gemini-3.8-flash")
PRO_MODEL = os.environ.get("PRO_MODEL", "gemini-3.8-pro")
MODEL = FAST_MODEL

_COMPLEX_REASONING_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|overlap|shared|differ|difference|contrast|"
    r"prioriti[sz]e|rank|unmet\s+need|cross[- ]?indication|synerg\w+|combination|"
    r"across\s+diseases|between\s+\w+\s+and|why\s+does|white[- ]?space)\b",
    re.IGNORECASE,
)

RESEARCH_SCOPING_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This Disease Characterisation Dossier is generated for early-stage "
    "Life Sciences research and target scoping only. It does not constitute clinical "
    "or medical advice.*"
)


async def initialize_session_state(callback_context: CallbackContext) -> None:
    """Initializes session state keys so template placeholders always resolve."""
    state = callback_context.state
    defaults = {
        "resolution_status": "pending",
        "efo_id": "Not yet resolved.",
        "canonical_disease_name": "Not yet resolved.",
        "icd10_codes_str": "Not yet resolved.",
        "disease_context": "Not yet resolved.",
        "targets_report": "Not yet retrieved.",
        "treatments_report": "Not yet retrieved.",
        "literature_report": "Not yet retrieved.",
        "characterized_diseases_index": {},
        "characterized_diseases_summary": "None characterized yet in this session.",
    }
    for key, default_val in defaults.items():
        if key not in state:
            state[key] = default_val


async def check_ontology_resolution(ctx: Context, node_input: Any) -> str:
    """Routes workflow execution based on whether disease ontology resolution succeeded, is ambiguous, or failed."""
    status = str(ctx.state.get("resolution_status") or "")
    efo_id = str(ctx.state.get("efo_id") or "")
    if status == "success" and efo_id and efo_id != "Not yet resolved.":
        ctx.route = "resolved"
    elif status == "ambiguous":
        ctx.route = "ambiguous"
    else:
        ctx.route = "unresolved"
    return str(node_input or "")


async def ambiguous_disease_handler(ctx: Context, node_input: Any) -> str:
    """Human-in-the-Loop disambiguation gate when a query matches multiple candidate indications."""
    candidates = ctx.state.get("ambiguous_candidates") or []
    rows = []
    for cand in candidates:
        if isinstance(cand, dict):
            c_name = cand.get("name") or "Unknown"
            c_efo = cand.get("efo_id") or "N/A"
            c_desc = (cand.get("description") or "No description available.").replace(
                "\n", " "
            )
            rows.append(f"| **{c_name}** | `{c_efo}` | {c_desc} |")

    table_md = ""
    if rows:
        table_md = (
            "\n\n| Candidate Indication | Canonical EFO / MONDO ID | Description |\n"
            "| :--- | :--- | :--- |\n"
            + "\n".join(rows)
        )

    summary = str(
        node_input
        or ctx.state.get("disease_context")
        or "Multiple candidate indications matched your query."
    )
    return (
        "### Human-in-the-Loop Disambiguation Required\n\n"
        f"{summary}"
        f"{table_md}\n\n"
        "**Next Step:** Please reply with the specific **Candidate Indication Name** or "
        "canonical **`EFO_*` / `MONDO_*` ID** from the table above that you would like to characterize."
        f"{RESEARCH_SCOPING_DISCLAIMER}"
    )


async def unresolved_disease_handler(ctx: Context, node_input: Any) -> str:
    """Short-circuits the pipeline when a disease cannot be resolved in Open Targets."""
    summary = str(
        node_input
        or ctx.state.get("disease_context")
        or "The requested disease could not be resolved."
    )
    return (
        "### Disease Ontology Resolution Unsuccessful\n\n"
        f"{summary}\n\n"
        "Please check the spelling of the indication, supply a recognized medical synonym, "
        f"or provide a canonical `EFO_*` / `MONDO_*` identifier.{RESEARCH_SCOPING_DISCLAIMER}"
    )


def _build_model_for_role(
    role: Literal["extraction", "synthesis", "coordinator"] = "extraction",
) -> Gemini:
    """Constructs a role-appropriate Gemini model instance for strategic model routing.

    - `extraction`: Uses `FAST_MODEL` (`gemini-3.8-flash`) for deterministic tool calling and structured extraction.
    - `synthesis`: Uses `PRO_MODEL` (`gemini-3.8-pro`) for multi-modal cross-domain dossier synthesis.
    - `coordinator`: Defaults to `FAST_MODEL` and is dynamically upgraded per turn by `route_model_by_complexity`.
    """
    selected_model = PRO_MODEL if role == "synthesis" else FAST_MODEL
    return Gemini(
        model=selected_model,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def _build_model() -> Gemini:
    """Default model builder alias returning the fast extraction model."""
    return _build_model_for_role("extraction")


def classify_query_complexity(
    user_query: str,
    characterized_diseases_count: int = 0,
) -> tuple[Literal["fast_extraction", "deep_synthesis"], str, str]:
    """Classifies user query complexity to select `(routing_tier, model_name, routing_reason)`."""
    cleaned = redact_pii((user_query or "").strip())
    word_count = len(cleaned.split())
    pattern_match = _COMPLEX_REASONING_PATTERN.search(cleaned)

    if pattern_match:
        matched_term = pattern_match.group(0).lower()
        return (
            "deep_synthesis",
            PRO_MODEL,
            f"comparative_or_analytical_intent:{matched_term}",
        )
    if characterized_diseases_count >= 2 and word_count >= 12:
        return (
            "deep_synthesis",
            PRO_MODEL,
            f"multi_disease_session_synthesis:count={characterized_diseases_count}",
        )
    if word_count >= 30:
        return (
            "deep_synthesis",
            PRO_MODEL,
            f"high_complexity_prompt:words={word_count}",
        )
    return (
        "fast_extraction",
        FAST_MODEL,
        "single_indication_dispatch_or_lookup",
    )


async def route_model_by_complexity(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """ADK `before_model_callback` that dynamically routes `llm_request.model` based on query complexity."""
    user_text = ""
    user_content = getattr(callback_context, "user_content", None)
    if user_content and getattr(user_content, "parts", None):
        user_text = " ".join(
            p.text for p in user_content.parts if getattr(p, "text", None)
        )
    elif llm_request.contents:
        for content in reversed(llm_request.contents):
            if getattr(content, "role", None) == "user" and getattr(
                content, "parts", None
            ):
                texts = [
                    p.text for p in content.parts if getattr(p, "text", None)
                ]
                if texts:
                    user_text = " ".join(texts)
                    break

    state = getattr(callback_context, "state", {}) or {}
    diseases_index = state.get("characterized_diseases_index")
    diseases_count = (
        len(diseases_index) if isinstance(diseases_index, dict) else 0
    )

    tier, selected_model, reason = classify_query_complexity(
        user_text,
        characterized_diseases_count=diseases_count,
    )
    llm_request.model = selected_model
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.temperature = 0.2 if tier == "deep_synthesis" else 0.0

    routing_record = {
        "tier": tier,
        "model": selected_model,
        "reason": reason,
    }
    try:
        callback_context.state["last_model_routing"] = routing_record
    except Exception:
        pass

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("gen_ai.model.routing_tier", tier)
        span.set_attribute("gen_ai.model.selected", selected_model)
        span.set_attribute("gen_ai.model.routing_reason", reason)

    logger.info(
        "Strategic model router selected %s (tier=%s, reason=%s)",
        selected_model,
        tier,
        reason,
        extra={
            "structured_data": {
                "event_type": "strategic_model_routing",
                "agent_name": getattr(callback_context, "agent_name", "root_agent"),
                "routing_tier": tier,
                "selected_model": selected_model,
                "routing_reason": reason,
            }
        },
    )
    return None


async def fallback_to_fast_model_on_error(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """ADK `on_model_error_callback` providing automatic resilience fallback from `PRO_MODEL` to `FAST_MODEL`."""
    failed_model = getattr(llm_request, "model", None) or PRO_MODEL
    if failed_model == FAST_MODEL:
        return None

    logger.warning(
        "Model '%s' encountered error (%s); automatically falling back to '%s'",
        failed_model,
        error,
        FAST_MODEL,
        extra={
            "structured_data": {
                "event_type": "model_routing_fallback",
                "failed_model": failed_model,
                "fallback_model": FAST_MODEL,
                "error": str(error),
            }
        },
    )
    llm_request.model = FAST_MODEL
    fallback_llm = Gemini(
        model=FAST_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )
    last_response: LlmResponse | None = None
    async for resp in fallback_llm.generate_content_async(
        llm_request, stream=False
    ):
        last_response = resp
    return last_response


disease_resolver_agent = Agent(
    name="disease_resolver_agent",
    model=_build_model_for_role("extraction"),
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    description="Resolves a user-entered disease into canonical EFO and ICD-10 ontology identifiers.",
    instruction="""You are the Disease Ontology Normalization Specialist.
Given the user's disease query:
1. Call `resolve_disease_ontology` with the disease name or acronym.
2. Output a concise, structured summary containing:
   - Canonical Disease Name
   - Canonical EFO / MONDO ID (`efo_id`)
   - ICD-10 Code(s) (`icd10_codes`; if empty, state 'ICD-10: Not mapped in ontology cross-references')
   - Disease Description, Synonyms, and Therapeutic Areas
   - Open Targets Disease URL
3. Recovery Instructions:
   - If the tool returns `status="validation_error"`, `status="not_found"`, `status="ambiguous"`, or `status="error"`, inspect the `recovery_instructions` field in the tool output and follow it explicitly.
   - If `status="validation_error"` can be fixed by normalizing the query (e.g., stripping stray punctuation), retry once with the cleaned query; otherwise report the `message` and `recovery_instructions` clearly without inventing EFO or ICD-10 codes.""",
    tools=[resolve_disease_ontology],
    output_key="disease_context",
)

targets_agent = Agent(
    name="targets_agent",
    model=_build_model_for_role("extraction"),
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    description="Fetches top known biological targets and tractability evidence for the resolved disease.",
    instruction="""You are the Target Discovery Specialist.
Structured Ontology Identifiers from Session State:
- Canonical EFO ID: `{efo_id}`
- Canonical Disease Name: `{canonical_disease_name}`
- ICD-10 Code(s): `{icd10_codes_str}`

Resolved Disease Context:
{disease_context}

1. Call `fetch_known_targets(efo_id="{efo_id}", limit=10)` using the exact canonical `efo_id` above.
2. Summarize the top associated biological targets including:
   - Gene Symbol and Ensembl ID (`ENSG...`)
   - Target Name
   - Overall Association Score and key datatype scores (genetic, somatic, known drug, literature, pathways)
   - Positive Tractability / Druggability modalities (Small Molecule, Antibody, PROTAC, etc.)
3. Recovery Instructions:
   - If `fetch_known_targets` returns `status="validation_error"`, verify `efo_id` uses underscore format (e.g., `EFO_0000768` rather than `EFO:0000768`) and `limit` is between 1 and 25, then retry once.
   - If `fetch_known_targets` returns `status="not_found"`, `status="error"`, or an empty `targets` list, follow the `recovery_instructions` field in the tool output and state clearly that no target associations were returned by Open Targets rather than fabricating target symbols, Ensembl IDs, or scores.""",
    tools=[fetch_known_targets],
    output_key="targets_report",
)

treatments_agent = Agent(
    name="treatments_agent",
    model=_build_model_for_role("extraction"),
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    description="Fetches existing approved/investigational treatments and active clinical trials for the disease.",
    instruction="""You are the Clinical & Therapeutics Landscape Specialist.
Structured Ontology Identifiers from Session State:
- Canonical EFO ID: `{efo_id}`
- Canonical Disease Name: `{canonical_disease_name}`
- ICD-10 Code(s): `{icd10_codes_str}`

Resolved Disease Context:
{disease_context}

1. Call `fetch_treatments_and_trials(efo_id="{efo_id}", disease_name="{canonical_disease_name}", limit=10)` using the exact canonical `efo_id` and `canonical_disease_name` above.
2. Summarize:
   - Approved and investigational drugs (`drug_name`, `chembl_id`, modality `drug_type`, `max_clinical_phase`, `mechanism_of_action`, and `target_symbol`)
   - Active or recent clinical trials from ClinicalTrials.gov (`nct_id`, `title`, `phases`, `overall_status`, `interventions`, and `lead_sponsor`)
3. Recovery Instructions:
   - If `fetch_treatments_and_trials` returns `status="validation_error"`, normalize `efo_id` to underscore format and ensure `disease_name` is non-empty and `limit` is between 1 and 25, then retry once.
   - If `warnings` or `recovery_instructions` are present in the tool output (for example, partial upstream degradation or zero drugs/trials for a rare indication), follow `recovery_instructions` and explicitly note the missing or degraded source without fabricating ChEMBL IDs, drug names, or NCT trial identifiers.""",
    tools=[fetch_treatments_and_trials],
    output_key="treatments_report",
)

literature_agent = Agent(
    name="literature_agent",
    model=_build_model_for_role("extraction"),
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    description="Surfaces up to 5 recent peer-reviewed publications and bioRxiv/medRxiv preprints.",
    instruction="""You are the Biomedical Literature & Preprint Specialist.
Structured Ontology Identifiers from Session State:
- Canonical EFO ID: `{efo_id}`
- Canonical Disease Name: `{canonical_disease_name}`

Resolved Disease Context:
{disease_context}

1. Call `fetch_recent_publications(disease_name="{canonical_disease_name}", max_results=5)` using the exact canonical disease name above.
2. Report AT MOST 5 publications (if the tool returns fewer than 5, report only those returned—never invent papers to reach 5).
3. For each publication, preserve:
   - Source badge: `[Peer-Reviewed]` or `[Preprint - bioRxiv]` / `[Preprint - medRxiv]`
   - Title, Authors, Venue/Server, and Publication Year
   - Identifier (`PMID` and/or `DOI`) and URL
   - Concise abstract takeaway.
4. Recovery Instructions:
   - If `fetch_recent_publications` returns `status="validation_error"`, ensure `disease_name` is non-empty and `max_results` is between 1 and 5, then retry once.
   - If `returned_count == 0` or `recovery_instructions` indicates a narrower subtype name returned no hits, retry once using the broader parent disease name or primary synonym from `{disease_context}`. If still empty or `status="error"`, report the unavailability transparently without fabricating PMIDs or DOIs.""",
    tools=[fetch_recent_publications],
    output_key="literature_report",
)

research_join = JoinNode(
    name="research_join",
    description="Synchronizes parallel target, treatment, and literature research branches.",
)

dossier_synthesizer_agent = Agent(
    name="dossier_synthesizer_agent",
    model=_build_model_for_role("synthesis"),
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    description="Synthesizes the grounded Disease Characterisation Dossier and saves it as a downloadable artifact.",
    instruction=f"""You are the Lead Life Sciences Scoping Synthesizer.
Combine the findings from the specialist agents into a clear, executive **Disease Characterisation Dossier**:

- **Structured Ontology Identifiers**:
  - Canonical Disease Name: `{{canonical_disease_name}}`
  - Canonical EFO ID: `{{efo_id}}`
  - ICD-10 Code(s): `{{icd10_codes_str}}`

- **Disease Ontology Context**:
{{disease_context}}

- **Known Targets Report**:
{{targets_report}}

- **Existing Treatments & Clinical Trials Report**:
{{treatments_report}}

- **Recent Publications & Preprints Report**:
{{literature_report}}

Structure your Markdown response with these exact sections:
1. **Disease Ontology & Normalization Header**:
   - Render a prominent summary table at the very top showing **Disease Name** (`{{canonical_disease_name}}`), **EFO ID** (`{{efo_id}}`), **ICD-10 Code(s)** (`{{icd10_codes_str}}`), **Therapeutic Areas**, and **Open Targets Link**.
2. **Executive Scoping Summary**:
   - 2-3 sentences summarizing the disease biology, target landscape, and therapeutic maturity.
3. **Known Biological Targets**:
   - Markdown table of top targets (`Gene Symbol`, `Ensembl ID`, `Target Name`, `Association Score`, `Key Evidence`, `Tractability`).
4. **Existing Treatments & Clinical Trial Pipeline**:
   - Markdown table of known drugs (`Drug Name`, `ChEMBL ID`, `Modality`, `Mechanism of Action`, `Target`, `Max Clinical Phase`) followed by notable clinical trials (`NCT ID`, `Title`, `Phase`, `Status`, `Intervention`).
5. **Recent Publications & Preprints (Top <= 5)**:
   - Numbered list of up to 5 publications (fewer if fewer exist). Each item MUST include its badge (`[Peer-Reviewed]`, `[Preprint - bioRxiv]`, or `[Preprint - medRxiv]`), `PMID`/`DOI` link, and a 1-2 sentence synthesis.
6. **Research Scoping Disclaimer**:
   - End with: {RESEARCH_SCOPING_DISCLAIMER.strip()}

Finally, call `save_dossier_artifact` with `efo_id="{{efo_id}}"` and the complete Markdown dossier text, and present the full Markdown dossier to the user.
Recovery Instructions:
- If `save_dossier_artifact` returns `status="validation_error"`, follow its `recovery_instructions` (ensure `efo_id` matches the canonical ID and `markdown_content` contains the full Markdown dossier of at least 20 characters) and retry once.
- If `save_dossier_artifact` returns `status="error"`, still output the complete Markdown dossier in your response and note that artifact persistence was skipped.""",
    tools=[save_dossier_artifact],
    output_key="characterisation_dossier",
    on_model_error_callback=fallback_to_fast_model_on_error,
    after_agent_callback=trigger_async_memory_consolidation,
)

disease_characterisation_pipeline = Workflow(
    name="disease_characterisation_pipeline",
    description="Executes full disease characterisation: ontology normalization (EFO + ICD-10), conditional short-circuit/HITL disambiguation check, parallel research across targets, treatments, and literature/preprints, and dossier synthesis.",
    input_schema=str,
    edges=[
        (
            "START",
            disease_resolver_agent,
            check_ontology_resolution,
            {
                "resolved": (targets_agent, treatments_agent, literature_agent),
                "ambiguous": ambiguous_disease_handler,
                "unresolved": unresolved_disease_handler,
            },
        ),
        (
            (targets_agent, treatments_agent, literature_agent),
            research_join,
            dossier_synthesizer_agent,
        ),
    ],
)

root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="disease_characterisation",
    model=_build_model_for_role("coordinator"),
    description="Automated Disease Characterisation Agent for Life Sciences scoping.",
    instruction="""You are the Automated Disease Characterisation Coordinator for Life Sciences scoping.

- When the user enters a disease name or asks to characterize/scope a disease, call the `disease_characterisation_pipeline` workflow tool (passing the disease name/query as `request`) to run the full ontology normalization (EFO + ICD-10), parallel target/treatment/literature analysis, and dossier synthesis. Once `disease_characterisation_pipeline` completes, present its synthesized Markdown dossier verbatim in your response without omitting any tables, identifiers, badges, or disclaimers.
- When the user asks a targeted follow-up question or a comparison across multiple diseases characterized in the current session or past sessions, answer using the session state (`{disease_context}`, `{targets_report}`, `{treatments_report}`, `{literature_report}`, and the consolidated multi-disease index below), or retrieve persistent cross-session knowledge using `search_long_term_disease_memory` / `load_memory` (or call the specific live biomedical tool `resolve_disease_ontology`, `fetch_known_targets`, `fetch_treatments_and_trials`, `fetch_recent_publications`, `save_dossier_artifact` as needed):

Consolidated Session Diseases Index:
{characterized_diseases_summary}

- Recovery Instructions: When calling any tool directly, inspect its `status` and `recovery_instructions` fields. If `status="validation_error"`, correct the input arguments according to `recovery_instructions` and retry once. If `status="ambiguous"` or `status="not_found"`, surface the `recovery_instructions` or candidate disambiguation table to the user rather than guessing identifiers.
- Always ground every identifier (`EFO`, `ICD-10`, `ENSG`, `ChEMBL`, `NCT`, `PMID`, `DOI`) in tool outputs, cap publication lists at a maximum of 5 (labeling `bioRxiv`/`medRxiv` preprints clearly), and include a non-clinical research scoping disclaimer.""",
    tools=[
        disease_characterisation_pipeline,
        resolve_disease_ontology,
        fetch_known_targets,
        fetch_treatments_and_trials,
        fetch_recent_publications,
        save_dossier_artifact,
        search_long_term_disease_memory,
        load_memory,
        preload_memory,
        *get_optional_vertex_ai_search_tools(),
    ],
    before_agent_callback=initialize_session_state,
    before_model_callback=route_model_by_complexity,
    on_model_error_callback=fallback_to_fast_model_on_error,
    after_agent_callback=trigger_async_memory_consolidation,
)

app = App(
    root_agent=root_agent,
    name="app",
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1,
    ),
    plugins=[
        DiseaseObservabilityPlugin(),
        ReflectAndRetryToolPlugin(),
    ],
)
