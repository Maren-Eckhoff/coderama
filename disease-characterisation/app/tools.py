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

"""Biomedical characterisation tools for the Disease Characterisation Agent."""

from __future__ import annotations

import asyncio
import re
from typing import Annotated, Any

import httpx
from google.adk.tools import ToolContext
from google.genai import types
from opentelemetry import trace
from pydantic import Field, ValidationError

from app.app_utils.telemetry import get_structured_logger, redact_pii
from app.schemas import (
    ONTOLOGY_ID_PATTERN,
    DiseaseOntologyOutput,
    FetchKnownTargetsInput,
    FetchRecentPublicationsInput,
    FetchTreatmentsAndTrialsInput,
    KnownTargetsOutput,
    RecentPublicationsOutput,
    ResolveDiseaseInput,
    SaveDossierArtifactInput,
    SaveDossierArtifactOutput,
    TreatmentsAndTrialsOutput,
    validate_and_dump,
)

logger = get_structured_logger("disease_characterisation.tools")
tracer = trace.get_tracer("disease_characterisation.tools")

OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
EBI_OLS4_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
CHEMBL_DRUG_INDICATION_URL = "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_MECHANISM_URL = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_PUBLICATIONS_CAP = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _async_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    """Executes an async HTTP request with exponential backoff on transient 429/5xx errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if method.upper() == "POST":
                resp = await client.post(url, **kwargs)
            else:
                resp = await client.get(url, **kwargs)
            if (
                resp.status_code in RETRYABLE_STATUS_CODES
                and attempt < max_attempts
            ):
                await asyncio.sleep(0.3 * (2 ** (attempt - 1)))
                continue
            return resp
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                await asyncio.sleep(0.3 * (2 ** (attempt - 1)))
            else:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        f"HTTP {method} request to {url} failed after {max_attempts} attempts."
    )


def _extract_icd10_codes(db_xrefs: list[str]) -> list[str]:
    """Extract normalized ICD-10 codes from Open Targets dbXRefs strings."""
    icd10_codes: list[str] = []
    seen: set[str] = set()
    for ref in db_xrefs:
        if not isinstance(ref, str):
            continue
        upper_ref = ref.strip().upper()
        if upper_ref.startswith(("ICD10:", "ICD10CM:", "ICD-10:", "ICD10WHO:")):
            code = ref.split(":", 1)[1].strip()
            formatted = f"ICD10:{code}"
            if formatted not in seen:
                seen.add(formatted)
                icd10_codes.append(formatted)
    return icd10_codes


async def _lookup_icd10_via_ols4(
    disease_name: str, efo_id: str, client: httpx.AsyncClient
) -> list[str]:
    """Fallback lookup of ICD-10 cross-references via the EBI OLS4 REST API."""
    icd10_codes: list[str] = []
    seen: set[str] = set()
    with tracer.start_as_current_span("ebi.ols4.icd10_fallback") as span:
        span.set_attribute("disease.efo_id", efo_id)
        try:
            resp = await _async_request_with_retry(
                client,
                "GET",
                EBI_OLS4_SEARCH_URL,
                params={
                    "q": efo_id if efo_id else disease_name,
                    "FieldList": "iri,label,short_form,obo_id,annotations_trimmed",
                    "rows": 5,
                },
            )
            span.set_attribute("http.status_code", resp.status_code)
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            for doc in docs:
                for text in doc.get("annotations_trimmed", []) or []:
                    for match in re.findall(
                        r"\bICD10(?:CM)?[:\s]+([A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?)\b",
                        str(text),
                        flags=re.IGNORECASE,
                    ):
                        code = f"ICD10:{match.upper()}"
                        if code not in seen:
                            seen.add(code)
                            icd10_codes.append(code)
            span.set_attribute("disease.icd10_fallback_count", len(icd10_codes))
        except Exception as exc:
            logger.debug("OLS4 ICD-10 fallback lookup failed: %s", exc)
    return icd10_codes


_DIRECT_ONTOLOGY_ID_RE = re.compile(
    r"^(EFO|MONDO|ORPHA|Orphanet|HP|OTAR|DOID)[:_](\d+)$", re.IGNORECASE
)
_DISEASE_SUFFIXES = (" disease", " syndrome", " disorder", " condition")
_ACRONYM_STOPWORDS = {"of", "the", "and", "in", "with", "or", "by", "for"}


def _normalize_disease_term(text: str) -> str:
    """Normalizes a disease string for deterministic comparison."""
    lowered = (text or "").lower().replace("’s", "").replace("'s", "")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def _strip_disease_suffixes(norm_text: str) -> str:
    """Strips generic disease/syndrome suffixes from a normalized term."""
    result = norm_text
    for suffix in _DISEASE_SUFFIXES:
        if result.endswith(suffix) and len(result) > len(suffix):
            result = result[: -len(suffix)].strip()
    return result


def _compute_acronym(text: str) -> str:
    """Computes standard initial-letter acronym from a disease name (including hyphenated parts)."""
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    filtered = [t for t in tokens if t not in _ACRONYM_STOPWORDS]
    return "".join(t[0] for t in filtered if t)


def _rank_disease_candidates(
    query: str, hits: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, bool]:
    """Re-ranks candidate disease hits and detects whether the query is ambiguous.

    Returns:
        tuple of (reordered_hits, resolution_confidence, is_ambiguous)
    """
    if not hits:
        return [], "not_found", False

    q_norm = _normalize_disease_term(query)
    q_core = _strip_disease_suffixes(q_norm)
    q_upper_id = query.strip().replace(":", "_").upper()

    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx, hit in enumerate(hits):
        hit_id = str(hit.get("id") or "")
        hit_name = str(hit.get("name") or "")
        h_norm = _normalize_disease_term(hit_name)
        h_core = _strip_disease_suffixes(h_norm)
        h_acronym = _compute_acronym(hit_name)

        synonym_terms: list[str] = []
        for syn in hit.get("synonyms") or []:
            if isinstance(syn, str):
                synonym_terms.append(syn)
            elif isinstance(syn, dict):
                synonym_terms.extend(syn.get("terms") or [])
        syn_norms = {_normalize_disease_term(s) for s in synonym_terms if s}
        syn_cores = {_strip_disease_suffixes(s) for s in syn_norms if s}

        if hit_id.upper() == q_upper_id:
            score, conf = 100, "direct_id"
        elif h_norm == q_norm or (q_core and h_core == q_core):
            score, conf = 95, "exact_name"
        elif (
            q_norm in syn_norms
            or (q_core and q_core in syn_cores)
            or (len(q_norm) >= 2 and h_acronym == q_norm)
        ):
            score, conf = 90, "exact_synonym"
        elif len(q_core.split()) >= 2 and all(
            tok in h_norm.split() for tok in q_core.split()
        ):
            score, conf = 85, "high_token_match"
        else:
            score, conf = 50 - idx, "fuzzy_search_hit"

        scored.append((score, -idx, conf, hit))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    reordered = [item[3] for item in scored]
    top_score, _, top_conf, _ = scored[0]

    has_tie = len(scored) >= 2 and scored[0][0] == scored[1][0] and top_score < 95
    is_ambiguous = len(reordered) >= 2 and (
        top_conf == "fuzzy_search_hit" or has_tie
    )
    return reordered, top_conf, is_ambiguous


def _record_unresolved_state(
    tool_context: ToolContext | None,
    status: str,
    query: str,
    payload: dict[str, Any],
) -> None:
    """Writes structured state keys when disease ontology resolution fails or returns no match."""
    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["resolution_status"] = status
        tool_context.state["efo_id"] = "Not yet resolved."
        tool_context.state["canonical_disease_name"] = query
        tool_context.state["icd10_codes"] = []
        tool_context.state["icd10_codes_str"] = (
            "ICD-10: Not mapped in ontology cross-references"
        )
        tool_context.state["disease_ontology_record"] = payload


async def resolve_disease_ontology(
    disease_query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=300,
            description=(
                "The disease name, medical acronym (e.g., 'IPF', 'NSCLC'), or canonical "
                "EFO/MONDO identifier (e.g., 'EFO_0000768', 'MONDO_0005148')."
            ),
        ),
    ],
    tool_context: ToolContext,
) -> DiseaseOntologyOutput:
    """Resolves a free-text disease query into canonical EFO and ICD-10 ontology identifiers.

    Supports direct canonical ID lookup (`EFO_*`, `MONDO_*`, `ORPHA_*`, `HP_*`), deterministic
    exact-name and synonym/acronym re-ranking, and Human-in-the-Loop (HITL) disambiguation
    when a broad or ambiguous query matches multiple distinct indications.

    Recovery Instructions:
        - If `status == "validation_error"`: Provide a non-empty `disease_query` (1-300 chars) and retry.
        - If `status == "not_found"`: Do NOT invent an EFO or ICD-10 code. Retry `resolve_disease_ontology`
          once using a broader parent indication name or standard medical synonym (removing subtype numerals
          or punctuation). If still `not_found`, report that the indication is unmapped in Open Targets.
        - If `status == "ambiguous"`: Do NOT invoke target, treatment, or literature tools yet. Present the
          returned `candidates` table (`name`, `efo_id`, `description`) and ask the user to confirm or select
          the desired `efo_id`.
        - If `status == "error"`: Report the temporary Open Targets API failure and follow `recovery_instructions`.

    Args:
        disease_query: The disease name, synonym, acronym, or canonical EFO/MONDO ID entered by the user.

    Returns:
        Strict `DiseaseOntologyOutput` dictionary containing `status`, `resolution_confidence`,
        `efo_id`, `name`, `icd10_codes`, `synonyms`, `therapeutic_areas`, `candidates`, and `recovery_instructions`.
    """
    raw_cleaned = redact_pii((disease_query or "").strip())
    try:
        validated_in = ResolveDiseaseInput(disease_query=raw_cleaned)
        cleaned_query = validated_in.disease_query
    except ValidationError as val_exc:
        err_res = validate_and_dump(
            DiseaseOntologyOutput,
            {
                "status": "validation_error",
                "resolution_confidence": "validation_error",
                "query": raw_cleaned,
                "message": f"Invalid disease_query input: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "Call `resolve_disease_ontology` with a non-empty disease name "
                    "(e.g., 'Idiopathic Pulmonary Fibrosis'), medical acronym ('IPF'), "
                    "or canonical ontology ID ('EFO_0000768')."
                ),
            },
        )
        _record_unresolved_state(tool_context, "error", raw_cleaned, err_res)
        return err_res  # type: ignore[return-value]

    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        parent_span.set_attribute("disease.query", cleaned_query)

    search_gql = """
    query SearchDisease($queryString: String!) {
      search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 5}) {
        hits {
          id
          name
          description
        }
      }
    }
    """

    details_gql = """
    query DiseaseDetails($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        description
        dbXRefs
        synonyms {
          relation
          terms
        }
        therapeuticAreas {
          id
          name
        }
      }
    }
    """

    tool_conf = (
        getattr(tool_context, "tool_confirmation", None)
        if tool_context is not None
        else None
    )
    confirmed_efo_override: str | None = None
    if tool_conf is not None:
        if not tool_conf.confirmed:
            declined_res = validate_and_dump(
                DiseaseOntologyOutput,
                {
                    "status": "ambiguous",
                    "resolution_confidence": "disambiguation_declined",
                    "query": cleaned_query,
                    "requires_human_disambiguation": True,
                    "message": (
                        f"Disambiguation for '{cleaned_query}' was declined. "
                        "Please provide a specific disease name or canonical EFO_*/MONDO_* identifier."
                    ),
                    "recovery_instructions": (
                        "Ask the user to supply the exact disease subtype name or canonical "
                        "`EFO_*` / `MONDO_*` ID they want to characterize."
                    ),
                },
            )
            _record_unresolved_state(
                tool_context, "ambiguous", cleaned_query, declined_res
            )
            return declined_res  # type: ignore[return-value]
        payload = tool_conf.payload if isinstance(tool_conf.payload, dict) else {}
        confirmed_efo_override = payload.get("selected_efo_id") or payload.get(
            "default_efo_id"
        )

    direct_match = _DIRECT_ONTOLOGY_ID_RE.match(cleaned_query)
    if direct_match:
        prefix = direct_match.group(1).upper()
        if prefix == "ORPHANET":
            prefix = "ORPHA"
        confirmed_efo_override = f"{prefix}_{direct_match.group(2)}"

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            hits: list[dict[str, Any]] = []
            resolution_confidence = "direct_id"
            is_ambiguous = False

            if confirmed_efo_override:
                efo_id = confirmed_efo_override
                top_hit: dict[str, Any] = {"id": efo_id, "name": cleaned_query}
                if tool_conf is not None and tool_conf.confirmed:
                    resolution_confidence = "human_confirmed"
            else:
                with tracer.start_as_current_span(
                    "opentargets.graphql.search_disease"
                ) as search_span:
                    search_span.set_attribute("graphql.operation", "SearchDisease")
                    search_span.set_attribute("disease.query", cleaned_query)
                    search_resp = await _async_request_with_retry(
                        client,
                        "POST",
                        OPEN_TARGETS_GRAPHQL_URL,
                        json={
                            "query": search_gql,
                            "variables": {"queryString": cleaned_query},
                        },
                    )
                    search_span.set_attribute(
                        "http.status_code", search_resp.status_code
                    )
                    search_resp.raise_for_status()
                    search_data = search_resp.json().get("data", {}) or {}
                    raw_hits = (search_data.get("search") or {}).get("hits") or []
                    search_span.set_attribute("search.hit_count", len(raw_hits))

                if not raw_hits:
                    not_found_res = validate_and_dump(
                        DiseaseOntologyOutput,
                        {
                            "status": "not_found",
                            "resolution_confidence": "not_found",
                            "query": cleaned_query,
                            "message": f"No canonical disease entry found in Open Targets for '{cleaned_query}'.",
                            "recovery_instructions": (
                                "Do NOT fabricate an EFO or ICD-10 identifier. If the query contained "
                                "abbreviations, typos, or narrow subtype qualifiers, retry `resolve_disease_ontology` "
                                "once with the standard medical name or broader parent disease term. Otherwise, "
                                "inform the user and request a canonical `EFO_*` or `MONDO_*` ID."
                            ),
                        },
                    )
                    _record_unresolved_state(
                        tool_context, "not_found", cleaned_query, not_found_res
                    )
                    return not_found_res  # type: ignore[return-value]

                hits, resolution_confidence, is_ambiguous = _rank_disease_candidates(
                    cleaned_query, raw_hits
                )
                top_hit = hits[0]
                efo_id = top_hit["id"]

            with tracer.start_as_current_span(
                "opentargets.graphql.disease_details"
            ) as details_span:
                details_span.set_attribute("graphql.operation", "DiseaseDetails")
                details_span.set_attribute("disease.efo_id", efo_id)
                details_resp = await _async_request_with_retry(
                    client,
                    "POST",
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={"query": details_gql, "variables": {"efoId": efo_id}},
                )
                details_span.set_attribute("http.status_code", details_resp.status_code)
                details_resp.raise_for_status()
                disease_obj = (
                    (details_resp.json().get("data") or {}).get("disease") or {}
                )

            if not disease_obj and confirmed_efo_override:
                not_found_res = validate_and_dump(
                    DiseaseOntologyOutput,
                    {
                        "status": "not_found",
                        "resolution_confidence": "not_found",
                        "query": cleaned_query,
                        "message": f"No canonical disease entry found in Open Targets for ID '{efo_id}'.",
                        "recovery_instructions": (
                            f"The ontology ID '{efo_id}' was not found in Open Targets. "
                            "Retry `resolve_disease_ontology` using the human-readable disease name instead."
                        ),
                    },
                )
                _record_unresolved_state(
                    tool_context, "not_found", cleaned_query, not_found_res
                )
                return not_found_res  # type: ignore[return-value]

            synonyms: list[str] = []
            for group in disease_obj.get("synonyms") or []:
                for term in group.get("terms") or []:
                    if term and term not in synonyms:
                        synonyms.append(term)

            if is_ambiguous and synonyms:
                q_norm = _normalize_disease_term(cleaned_query)
                q_core = _strip_disease_suffixes(q_norm)
                syn_norms = {_normalize_disease_term(s) for s in synonyms}
                syn_cores = {_strip_disease_suffixes(s) for s in syn_norms}
                if q_norm in syn_norms or (q_core and q_core in syn_cores):
                    resolution_confidence = "exact_synonym"
                    is_ambiguous = False

            db_xrefs = disease_obj.get("dbXRefs") or []
            icd10_codes = _extract_icd10_codes(db_xrefs)
            ols4_fallback_used = False
            if not icd10_codes:
                ols4_fallback_used = True
                icd10_codes = await _lookup_icd10_via_ols4(
                    disease_obj.get("name") or top_hit.get("name") or cleaned_query,
                    efo_id,
                    client,
                )

            therapeutic_areas = [
                {"id": ta.get("id"), "name": str(ta.get("name"))}
                for ta in (disease_obj.get("therapeuticAreas") or [])
                if ta.get("name")
            ]

            alternative_matches = [
                {
                    "efo_id": str(h.get("id")),
                    "name": str(h.get("name") or ""),
                    "description": str(h.get("description") or "")[:160],
                }
                for h in hits[1:5]
                if h.get("id")
            ]

            canonical_name = str(
                disease_obj.get("name") or top_hit.get("name") or cleaned_query
            )
            icd10_codes_str = (
                ", ".join(icd10_codes)
                if icd10_codes
                else "ICD-10: Not mapped in ontology cross-references"
            )

            if is_ambiguous:
                candidate_options = [
                    {
                        "efo_id": efo_id,
                        "name": canonical_name,
                        "description": str(
                            disease_obj.get("description")
                            or top_hit.get("description")
                            or ""
                        )[:160],
                    },
                    *alternative_matches,
                ]
                hint_msg = (
                    f"The disease query '{cleaned_query}' is ambiguous and matches "
                    f"{len(candidate_options)} distinct indications. Please confirm "
                    f"'{canonical_name}' ({efo_id}) or specify one of the candidate EFO IDs."
                )
                if (
                    tool_context is not None
                    and getattr(tool_context, "function_call_id", None)
                    and hasattr(tool_context, "request_confirmation")
                ):
                    tool_context.request_confirmation(
                        hint=hint_msg,
                        payload={
                            "query": cleaned_query,
                            "default_efo_id": efo_id,
                            "default_name": canonical_name,
                            "candidates": candidate_options,
                        },
                    )

                ambiguous_result = validate_and_dump(
                    DiseaseOntologyOutput,
                    {
                        "status": "ambiguous",
                        "resolution_confidence": "ambiguous",
                        "requires_human_disambiguation": True,
                        "query": cleaned_query,
                        "efo_id": efo_id,
                        "name": canonical_name,
                        "icd10_codes": icd10_codes,
                        "candidates": candidate_options,
                        "alternative_matches": alternative_matches,
                        "message": hint_msg,
                        "open_targets_url": f"https://platform.opentargets.org/disease/{efo_id}",
                        "recovery_instructions": (
                            "Present all entries in `candidates` (Disease Name, EFO ID, and Description) "
                            "to the user and ask them to confirm which specific EFO ID or subtype they "
                            "want characterized before running target, treatment, or literature lookups."
                        ),
                    },
                )
                if parent_span and parent_span.is_recording():
                    parent_span.set_attribute("disease.efo_id", efo_id)
                    parent_span.set_attribute(
                        "disease.resolution_confidence", "ambiguous"
                    )
                if tool_context is not None and hasattr(tool_context, "state"):
                    tool_context.state["resolution_status"] = "ambiguous"
                    tool_context.state["resolution_confidence"] = "ambiguous"
                    tool_context.state["ambiguous_candidates"] = candidate_options
                    tool_context.state["disease_context"] = ambiguous_result
                    tool_context.state["disease_ontology_record"] = ambiguous_result
                    tool_context.state["efo_id"] = efo_id
                    tool_context.state["canonical_disease_name"] = canonical_name
                    tool_context.state["icd10_codes"] = icd10_codes
                    tool_context.state["icd10_codes_str"] = icd10_codes_str
                return ambiguous_result  # type: ignore[return-value]

            recovery_note = (
                None
                if icd10_codes
                else (
                    "No ICD-10 cross-reference was mapped in Open Targets or OLS4 for this indication. "
                    "Explicitly display 'ICD-10: Not mapped in ontology cross-references' in the dossier header."
                )
            )
            result = validate_and_dump(
                DiseaseOntologyOutput,
                {
                    "status": "success",
                    "resolution_confidence": resolution_confidence,
                    "requires_human_disambiguation": False,
                    "query": cleaned_query,
                    "efo_id": efo_id,
                    "name": canonical_name,
                    "icd10_codes": icd10_codes,
                    "other_xrefs": [
                        ref
                        for ref in db_xrefs[:15]
                        if isinstance(ref, str)
                        and not ref.upper().startswith(("ICD10:", "ICD10CM:"))
                    ],
                    "description": str(
                        disease_obj.get("description")
                        or top_hit.get("description")
                        or ""
                    ),
                    "synonyms": synonyms[:10],
                    "therapeutic_areas": therapeutic_areas,
                    "alternative_matches": alternative_matches,
                    "open_targets_url": f"https://platform.opentargets.org/disease/{efo_id}",
                    "recovery_instructions": recovery_note,
                },
            )

            if parent_span and parent_span.is_recording():
                parent_span.set_attribute("disease.efo_id", efo_id)
                parent_span.set_attribute("disease.name", canonical_name)
                parent_span.set_attribute("disease.icd10_codes", ",".join(icd10_codes))
                parent_span.set_attribute(
                    "disease.resolution_confidence", resolution_confidence
                )
                parent_span.set_attribute(
                    "disease.ols4_fallback_used", ols4_fallback_used
                )

            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["resolution_status"] = "success"
                tool_context.state["resolution_confidence"] = resolution_confidence
                tool_context.state["disease_context"] = result
                tool_context.state["disease_ontology_record"] = result
                tool_context.state["efo_id"] = efo_id
                tool_context.state["canonical_disease_name"] = canonical_name
                tool_context.state["icd10_codes"] = icd10_codes
                tool_context.state["icd10_codes_str"] = icd10_codes_str

            return result  # type: ignore[return-value]
    except Exception as exc:
        logger.exception("Error resolving disease ontology for '%s'", cleaned_query)
        err_payload = validate_and_dump(
            DiseaseOntologyOutput,
            {
                "status": "error",
                "resolution_confidence": "error",
                "query": cleaned_query,
                "message": f"Failed to resolve disease ontology: {exc}",
                "recovery_instructions": (
                    "Open Targets GraphQL endpoint experienced a transient failure after 3 retries. "
                    "Do NOT hallucinate EFO or ICD-10 identifiers; report the upstream API error "
                    "to the user and suggest retrying."
                ),
            },
        )
        _record_unresolved_state(
            tool_context, "error", cleaned_query, err_payload
        )
        return err_payload  # type: ignore[return-value]


async def fetch_known_targets(
    efo_id: Annotated[
        str,
        Field(
            min_length=3,
            max_length=64,
            pattern=ONTOLOGY_ID_PATTERN,
            description="Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').",
        ),
    ],
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=15,
            description="Maximum number of top associated targets to return (1 to 15).",
        ),
    ] = 10,
) -> KnownTargetsOutput:
    """Fetches top-ranked known biological targets associated with a disease EFO/MONDO ID from Open Targets.

    Recovery Instructions:
        - If `status == "validation_error"`: Ensure `efo_id` matches `^(EFO|MONDO|ORPHA|HP|OTAR|DOID)_[0-9]+$`
          (never pass a free-text disease name as `efo_id`). Call `resolve_disease_ontology` first if needed.
        - If `status == "not_found"` or `targets` is empty: Do NOT fabricate target symbols or Ensembl IDs.
          Explicitly state in the dossier that Open Targets currently catalogs 0 target associations for this ID.
        - If `status == "error"`: Report that the Open Targets target association service experienced a transient
          error and omit unverified target claims.

    Args:
        efo_id: Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').
        limit: Maximum number of top targets to return (capped between 1 and 15).

    Returns:
        Strict `KnownTargetsOutput` dictionary with `status`, `efo_id`, `total_associated_targets`,
        `targets` list, and `recovery_instructions`.
    """
    raw_efo = redact_pii((efo_id or "").strip())
    try:
        validated_in = FetchKnownTargetsInput(efo_id=raw_efo, limit=limit)
        cleaned_efo = validated_in.efo_id
        capped_limit = validated_in.limit
    except ValidationError as val_exc:
        return validate_and_dump(  # type: ignore[return-value]
            KnownTargetsOutput,
            {
                "status": "validation_error",
                "efo_id": raw_efo,
                "message": f"Invalid parameters for fetch_known_targets: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "Pass a valid canonical ontology ID matching `^(EFO|MONDO|ORPHA|HP|OTAR|DOID)_[0-9]+$` "
                    "(e.g., 'EFO_0000768') and a positive integer `limit` between 1 and 15. "
                    "If you only have a disease name, call `resolve_disease_ontology` first."
                ),
            },
        )

    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        parent_span.set_attribute("disease.efo_id", cleaned_efo)

    targets_gql = """
    query AssociatedTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
          count
          rows {
            score
            target {
              id
              approvedSymbol
              approvedName
              biotype
              tractability {
                label
                modality
                value
              }
            }
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            with tracer.start_as_current_span(
                "opentargets.graphql.associated_targets"
            ) as ot_span:
                ot_span.set_attribute("graphql.operation", "AssociatedTargets")
                ot_span.set_attribute("disease.efo_id", cleaned_efo)
                resp = await _async_request_with_retry(
                    client,
                    "POST",
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={
                        "query": targets_gql,
                        "variables": {"efoId": cleaned_efo, "size": capped_limit},
                    },
                )
                ot_span.set_attribute("http.status_code", resp.status_code)
                resp.raise_for_status()
                disease_data = (resp.json().get("data") or {}).get("disease")

            if not disease_data:
                return validate_and_dump(  # type: ignore[return-value]
                    KnownTargetsOutput,
                    {
                        "status": "not_found",
                        "efo_id": cleaned_efo,
                        "message": f"No disease found for EFO ID '{cleaned_efo}'.",
                        "recovery_instructions": (
                            f"Verify the canonical EFO/MONDO ID via `resolve_disease_ontology` before retrying "
                            f"`fetch_known_targets`. Do not invent target associations."
                        ),
                    },
                )

            assoc = disease_data.get("associatedTargets") or {}
            rows = assoc.get("rows") or []
            targets_list: list[dict[str, Any]] = []

            for row in rows:
                target_info = row.get("target") or {}
                ensembl_id = str(target_info.get("id") or "")
                tractability_items = [
                    f"{t.get('modality')}: {t.get('label')}"
                    for t in (target_info.get("tractability") or [])
                    if t.get("value") is True
                ]
                datatype_scores = {
                    str(dt.get("id")): round(float(dt.get("score", 0.0)), 4)
                    for dt in (row.get("datatypeScores") or [])
                    if dt.get("id") is not None
                }
                targets_list.append(
                    {
                        "gene_symbol": target_info.get("approvedSymbol"),
                        "ensembl_id": ensembl_id,
                        "target_name": target_info.get("approvedName"),
                        "biotype": target_info.get("biotype"),
                        "overall_association_score": round(
                            float(row.get("score", 0.0)), 4
                        ),
                        "datatype_scores": datatype_scores,
                        "tractability_positive_modalities": tractability_items[:6],
                        "open_targets_url": f"https://platform.opentargets.org/target/{ensembl_id}",
                    }
                )

            total_count = int(assoc.get("count", len(targets_list)) or 0)
            if parent_span and parent_span.is_recording():
                parent_span.set_attribute("targets.total_count", total_count)
                parent_span.set_attribute("targets.returned_count", len(targets_list))

            empty_recovery = (
                "Open Targets returned 0 associated targets for this indication. "
                "State explicitly in the dossier that no validated biological targets are currently "
                "cataloged in Open Targets—never fabricate gene symbols or Ensembl IDs."
                if not targets_list
                else None
            )
            return validate_and_dump(  # type: ignore[return-value]
                KnownTargetsOutput,
                {
                    "status": "success",
                    "efo_id": cleaned_efo,
                    "disease_name": disease_data.get("name"),
                    "total_associated_targets": total_count,
                    "targets": targets_list,
                    "recovery_instructions": empty_recovery,
                },
            )
    except Exception as exc:
        logger.exception("Error fetching known targets for '%s'", cleaned_efo)
        return validate_and_dump(  # type: ignore[return-value]
            KnownTargetsOutput,
            {
                "status": "error",
                "efo_id": cleaned_efo,
                "message": f"Failed to fetch known targets: {exc}",
                "recovery_instructions": (
                    "Open Targets `associatedTargets` query failed after automatic retries. "
                    "Report the upstream service error in the Known Biological Targets section "
                    "and do NOT fabricate gene symbols or Ensembl IDs."
                ),
            },
        )


async def _fetch_drugs_via_chembl(
    efo_id: str, limit: int, client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    """Fallback lookup for approved/clinical drugs via EBI ChEMBL REST API."""
    colon_efo = efo_id.replace("_", ":")
    with tracer.start_as_current_span("ebi.chembl.fallback") as span:
        span.set_attribute("disease.efo_id", efo_id)
        resp = await _async_request_with_retry(
            client,
            "GET",
            CHEMBL_DRUG_INDICATION_URL,
            params={
                "efo_id": colon_efo,
                "limit": limit * 3,
                "format": "json",
            },
        )
        span.set_attribute("http.status_code", resp.status_code)
        resp.raise_for_status()
        indications = resp.json().get("drug_indications") or []
        if not indications:
            span.set_attribute("chembl.drugs_returned", 0)
            return []

        indications.sort(
            key=lambda x: float(x.get("max_phase_for_ind") or 0),
            reverse=True,
        )

        selected_chembl_ids: list[str] = []
        phase_by_chembl: dict[str, Any] = {}
        for ind in indications:
            mol_id = ind.get("molecule_chembl_id") or ind.get(
                "parent_molecule_chembl_id"
            )
            if mol_id and mol_id not in phase_by_chembl:
                phase_by_chembl[mol_id] = ind.get("max_phase_for_ind")
                selected_chembl_ids.append(mol_id)
            if len(selected_chembl_ids) >= limit:
                break

        if not selected_chembl_ids:
            span.set_attribute("chembl.drugs_returned", 0)
            return []

        ids_csv = ",".join(selected_chembl_ids)
        mol_map: dict[str, dict[str, Any]] = {}
        mech_map: dict[str, dict[str, Any]] = {}

        async def _load_molecules() -> None:
            try:
                mol_resp = await _async_request_with_retry(
                    client,
                    "GET",
                    CHEMBL_MOLECULE_URL,
                    params={
                        "molecule_chembl_id__in": ids_csv,
                        "limit": len(selected_chembl_ids),
                        "format": "json",
                    },
                )
                mol_resp.raise_for_status()
                for mol in mol_resp.json().get("molecules") or []:
                    cid = mol.get("molecule_chembl_id")
                    if cid:
                        mol_map[cid] = mol
            except Exception as exc:
                logger.debug("ChEMBL molecule metadata lookup failed: %s", exc)

        async def _load_mechanisms() -> None:
            try:
                mech_resp = await _async_request_with_retry(
                    client,
                    "GET",
                    CHEMBL_MECHANISM_URL,
                    params={
                        "molecule_chembl_id__in": ids_csv,
                        "limit": len(selected_chembl_ids) * 2,
                        "format": "json",
                    },
                )
                mech_resp.raise_for_status()
                for mech in mech_resp.json().get("mechanisms") or []:
                    cid = mech.get("molecule_chembl_id")
                    if cid and cid not in mech_map:
                        mech_map[cid] = mech
            except Exception as exc:
                logger.debug("ChEMBL mechanism lookup failed: %s", exc)

        await asyncio.gather(_load_molecules(), _load_mechanisms())

        results: list[dict[str, Any]] = []
        for cid in selected_chembl_ids:
            mol = mol_map.get(cid, {})
            mech = mech_map.get(cid, {})
            phase = phase_by_chembl.get(cid)
            results.append(
                {
                    "drug_name": mol.get("pref_name") or cid,
                    "chembl_id": str(cid),
                    "drug_type": mol.get("molecule_type"),
                    "max_clinical_phase": phase,
                    "trial_status": "Approved"
                    if str(phase) in ("4", "4.0")
                    else "Clinical Trial",
                    "mechanism_of_action": mech.get("mechanism_of_action"),
                    "target_symbol": mech.get("target_chembl_id"),
                    "target_ensembl_id": None,
                    "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/",
                }
            )
        span.set_attribute("chembl.drugs_returned", len(results))
        return results


async def fetch_treatments_and_trials(
    efo_id: Annotated[
        str,
        Field(
            min_length=3,
            max_length=64,
            pattern=ONTOLOGY_ID_PATTERN,
            description="Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').",
        ),
    ],
    disease_name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=300,
            description="Canonical disease name for querying ClinicalTrials.gov (e.g., 'Idiopathic Pulmonary Fibrosis').",
        ),
    ],
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=15,
            description="Maximum number of drugs and clinical trials to return (1 to 15).",
        ),
    ] = 10,
) -> TreatmentsAndTrialsOutput:
    """Fetches existing approved/investigational drugs from Open Targets / ChEMBL and clinical trials from ClinicalTrials.gov v2.

    Recovery Instructions:
        - If `status == "validation_error"`: Ensure `efo_id` matches `^(EFO|MONDO|ORPHA|HP|OTAR|DOID)_[0-9]+$`
          and `disease_name` is a non-empty canonical disease name (at least 2 chars), then retry.
        - If `partial_warnings` is non-empty: Synthesize the available `known_drugs` or `clinical_trials` that
          succeeded and note the partial source warning without inventing missing ChEMBL or NCT identifiers.
        - If both `known_drugs` and `clinical_trials` are empty: State explicitly that no approved/clinical drugs
          or active studies were found in Open Targets, ChEMBL, or ClinicalTrials.gov—never fabricate drug names or NCT IDs.

    Args:
        efo_id: Canonical ontology identifier (e.g., 'EFO_0000768').
        disease_name: Canonical disease name for querying ClinicalTrials.gov (e.g., 'Idiopathic Pulmonary Fibrosis').
        limit: Maximum number of drugs and clinical trials to return (capped between 1 and 15).

    Returns:
        Strict `TreatmentsAndTrialsOutput` dictionary with `status`, `efo_id`, `known_drugs`,
        `clinical_trials`, `partial_warnings`, and `recovery_instructions`.
    """
    raw_efo = redact_pii((efo_id or "").strip())
    raw_name = redact_pii((disease_name or "").strip())
    try:
        validated_in = FetchTreatmentsAndTrialsInput(
            efo_id=raw_efo,
            disease_name=raw_name,
            limit=limit,
        )
        cleaned_efo = validated_in.efo_id
        cleaned_name = validated_in.disease_name
        capped_limit = validated_in.limit
    except ValidationError as val_exc:
        return validate_and_dump(  # type: ignore[return-value]
            TreatmentsAndTrialsOutput,
            {
                "status": "validation_error",
                "efo_id": raw_efo,
                "disease_name": raw_name,
                "message": f"Invalid parameters for fetch_treatments_and_trials: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "Ensure `efo_id` is a valid canonical ID matching `^(EFO|MONDO|ORPHA|HP|OTAR|DOID)_[0-9]+$` "
                    "(e.g., 'EFO_0000768'), `disease_name` is at least 2 characters, and `limit` is between 1 and 15."
                ),
            },
        )

    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        parent_span.set_attribute("disease.efo_id", cleaned_efo)
        parent_span.set_attribute("disease.name", cleaned_name)

    drugs_gql = """
    query KnownDrugs($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        knownDrugs(size: $size) {
          uniqueDrugs
          uniqueTargets
          count
          rows {
            phase
            status
            drugId
            prefName
            drugType
            mechanismOfAction
            targetId
            target {
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """

    known_drugs: list[dict[str, Any]] = []
    unique_drugs_count = 0
    clinical_trials: list[dict[str, Any]] = []
    errors: list[str] = []
    chembl_fallback_used = False

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:

        async def _fetch_drugs_branch() -> None:
            nonlocal known_drugs, unique_drugs_count, chembl_fallback_used
            if not cleaned_efo:
                return
            try:
                with tracer.start_as_current_span(
                    "opentargets.graphql.known_drugs"
                ) as kd_span:
                    kd_span.set_attribute("graphql.operation", "KnownDrugs")
                    kd_span.set_attribute("disease.efo_id", cleaned_efo)
                    resp = await _async_request_with_retry(
                        client,
                        "POST",
                        OPEN_TARGETS_GRAPHQL_URL,
                        json={
                            "query": drugs_gql,
                            "variables": {
                                "efoId": cleaned_efo,
                                "size": capped_limit * 3,
                            },
                        },
                    )
                    kd_span.set_attribute("http.status_code", resp.status_code)
                    resp.raise_for_status()
                    disease_data = (
                        (resp.json().get("data") or {}).get("disease") or {}
                    )
                    kd = disease_data.get("knownDrugs") or {}
                    unique_drugs_count = int(kd.get("uniqueDrugs", 0) or 0)

                    seen_drug_targets: set[tuple[str, str]] = set()
                    for row in kd.get("rows") or []:
                        drug_id = str(row.get("drugId") or "")
                        target_id = str(row.get("targetId") or "")
                        key = (drug_id, target_id)
                        if key in seen_drug_targets:
                            continue
                        seen_drug_targets.add(key)
                        target_obj = row.get("target") or {}
                        known_drugs.append(
                            {
                                "drug_name": row.get("prefName"),
                                "chembl_id": drug_id,
                                "drug_type": row.get("drugType"),
                                "max_clinical_phase": row.get("phase"),
                                "trial_status": row.get("status"),
                                "mechanism_of_action": row.get("mechanismOfAction"),
                                "target_symbol": target_obj.get("approvedSymbol"),
                                "target_ensembl_id": target_id or None,
                                "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{drug_id}/"
                                if drug_id
                                else None,
                            }
                        )
                        if len(known_drugs) >= capped_limit:
                            break
                    kd_span.set_attribute("treatments.drugs_count", len(known_drugs))
            except Exception as exc:
                chembl_fallback_used = True
                logger.info(
                    "Open Targets knownDrugs fallback to ChEMBL for %s: %s",
                    cleaned_efo,
                    exc,
                    extra={
                        "structured_data": {
                            "event_type": "chembl_fallback",
                            "efo_id": cleaned_efo,
                        }
                    },
                )
                try:
                    known_drugs = await _fetch_drugs_via_chembl(
                        cleaned_efo, capped_limit, client
                    )
                    unique_drugs_count = len(known_drugs)
                except Exception as chembl_exc:
                    logger.warning("ChEMBL fallback query failed: %s", chembl_exc)
                    errors.append(f"Known drugs lookup error: {chembl_exc}")

        async def _fetch_trials_branch() -> None:
            if not cleaned_name:
                return
            try:
                with tracer.start_as_current_span(
                    "clinicaltrials.gov.v2.studies"
                ) as ct_span:
                    ct_span.set_attribute("disease.name", cleaned_name)
                    ct_resp = await _async_request_with_retry(
                        client,
                        "GET",
                        CLINICAL_TRIALS_API_URL,
                        params={
                            "query.cond": cleaned_name,
                            "pageSize": min(capped_limit, 10),
                            "format": "json",
                        },
                    )
                    ct_span.set_attribute("http.status_code", ct_resp.status_code)
                    ct_resp.raise_for_status()
                    studies = ct_resp.json().get("studies") or []
                    for study in studies[:capped_limit]:
                        proto = study.get("protocolSection") or {}
                        id_mod = proto.get("identificationModule") or {}
                        status_mod = proto.get("statusModule") or {}
                        design_mod = proto.get("designModule") or {}
                        arms_mod = proto.get("armsInterventionsModule") or {}
                        sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}

                        nct_id = str(id_mod.get("nctId") or "")
                        interventions = [
                            f"{iv.get('type', 'OTHER')}: {iv.get('name')}"
                            for iv in (arms_mod.get("interventions") or [])
                            if iv.get("name")
                        ]
                        clinical_trials.append(
                            {
                                "nct_id": nct_id,
                                "title": id_mod.get("briefTitle"),
                                "overall_status": status_mod.get("overallStatus"),
                                "phases": [
                                    str(p) for p in (design_mod.get("phases") or [])
                                ],
                                "interventions": interventions[:5],
                                "lead_sponsor": (
                                    sponsor_mod.get("leadSponsor") or {}
                                ).get("name"),
                                "url": f"https://clinicaltrials.gov/study/{nct_id}"
                                if nct_id
                                else None,
                            }
                        )
                    ct_span.set_attribute(
                        "treatments.trials_count", len(clinical_trials)
                    )
            except Exception as exc:
                logger.warning("ClinicalTrials.gov API v2 query failed: %s", exc)
                errors.append(f"ClinicalTrials.gov error: {exc}")

        await asyncio.gather(_fetch_drugs_branch(), _fetch_trials_branch())

    if parent_span and parent_span.is_recording():
        parent_span.set_attribute(
            "treatments.chembl_fallback_used", chembl_fallback_used
        )
        parent_span.set_attribute("treatments.drugs_count", len(known_drugs))
        parent_span.set_attribute("treatments.trials_count", len(clinical_trials))

    if not known_drugs and not clinical_trials and errors:
        return validate_and_dump(  # type: ignore[return-value]
            TreatmentsAndTrialsOutput,
            {
                "status": "error",
                "efo_id": cleaned_efo,
                "disease_name": cleaned_name,
                "partial_warnings": errors,
                "message": "; ".join(errors),
                "recovery_instructions": (
                    "Both drug and clinical trial lookups failed after retries. "
                    "Do NOT fabricate ChEMBL drug IDs or ClinicalTrials.gov NCT IDs; "
                    "report the upstream API unavailability in the dossier."
                ),
            },
        )

    recovery_msg: str | None = None
    if errors:
        recovery_msg = (
            "One upstream therapeutics endpoint returned a partial warning (`partial_warnings`). "
            "Report all successfully retrieved records and note the partial source limitation "
            "without inventing missing ChEMBL or NCT identifiers."
        )
    elif not known_drugs and not clinical_trials:
        recovery_msg = (
            "No approved/investigational drugs or clinical trials were found for this indication. "
            "Explicitly state that 0 drugs and 0 trials are currently cataloged—never fabricate records."
        )

    return validate_and_dump(  # type: ignore[return-value]
        TreatmentsAndTrialsOutput,
        {
            "status": "success",
            "efo_id": cleaned_efo,
            "disease_name": cleaned_name,
            "unique_known_drugs_count": unique_drugs_count,
            "known_drugs": known_drugs,
            "clinical_trials": clinical_trials,
            "partial_warnings": errors,
            "recovery_instructions": recovery_msg,
        },
    )


def _format_europe_pmc_record(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Europe PMC search result item with preprint vs. peer-reviewed badges."""
    source = (item.get("source") or "").upper()
    journal_title = item.get("journalTitle") or ""
    publisher = (
        (item.get("bookOrReportDetails") or {}).get("publisher")
        or item.get("publisher")
        or ""
    )
    doi = item.get("doi") or ""
    pmid = item.get("pmid") or ""

    combined_pub_info = f"{journal_title} {publisher}".lower()
    is_preprint = (
        source == "PPR"
        or "biorxiv" in combined_pub_info
        or "medrxiv" in combined_pub_info
    )
    if "biorxiv" in combined_pub_info or doi.startswith("10.1101/"):
        if "medrxiv" in combined_pub_info:
            preprint_server = "medRxiv"
        elif "biorxiv" in combined_pub_info:
            preprint_server = "bioRxiv"
        else:
            preprint_server = "bioRxiv/medRxiv"
    elif "medrxiv" in combined_pub_info:
        preprint_server = "medRxiv"
    else:
        preprint_server = publisher or journal_title or "Preprint Server"

    if is_preprint:
        publication_type = f"Preprint ({preprint_server})"
        source_badge = f"[Preprint - {preprint_server}]"
        venue = preprint_server
    else:
        publication_type = "Peer-Reviewed Journal Article"
        source_badge = "[Peer-Reviewed]"
        venue = journal_title or "Peer-Reviewed Journal"

    if doi:
        url = f"https://doi.org/{doi}"
    elif pmid:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    else:
        url = f"https://europepmc.org/article/{source}/{item.get('id', '')}"

    abstract_raw = item.get("abstractText") or ""
    abstract_clean = re.sub(r"<[^>]+>", "", abstract_raw).strip()
    if len(abstract_clean) > 450:
        abstract_clean = abstract_clean[:447] + "..."

    return {
        "title": item.get("title"),
        "authors": item.get("authorString"),
        "venue": str(venue),
        "publication_year": str(item.get("pubYear"))
        if item.get("pubYear") is not None
        else None,
        "first_publication_date": item.get("firstPublicationDate"),
        "publication_type": publication_type,
        "source_badge": source_badge,
        "is_preprint": is_preprint,
        "pmid": str(pmid) if pmid else None,
        "doi": str(doi) if doi else None,
        "cited_by_count": int(item.get("citedByCount") or 0),
        "url": url,
        "abstract_snippet": abstract_clean,
    }


def _is_allowed_preprint_item(item: dict[str, Any]) -> bool:
    """Checks whether a PPR item is from bioRxiv or medRxiv."""
    pub_text = (
        f"{item.get('journalTitle', '')} "
        f"{(item.get('bookOrReportDetails') or {}).get('publisher', '')} "
        f"{item.get('doi', '')}"
    ).lower()
    return any(
        tok in pub_text
        for tok in (
            "biorxiv",
            "medrxiv",
            "cold spring harbor",
            "10.1101/",
        )
    )


async def fetch_recent_publications(
    disease_name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=300,
            description="Canonical disease name to search in Europe PMC (PubMed + bioRxiv/medRxiv).",
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=5,
            description="Maximum number of publications to return (strictly capped at <= 5).",
        ),
    ] = 5,
) -> RecentPublicationsOutput:
    """Fetches up to 5 recent peer-reviewed publications (PubMed) and preprints (bioRxiv/medRxiv) for a disease.

    Executes concurrent queries for peer-reviewed articles (SRC:MED) and bioRxiv/medRxiv preprints (SRC:PPR)
    and merges them into a balanced list strictly capped at <= 5 publications.

    Recovery Instructions:
        - If `status == "validation_error"`: Provide a non-empty `disease_name` (>= 2 chars) and `max_results` between 1 and 5.
        - If `returned_count < 5` (including `0`): Report ONLY the publications returned in `publications`.
          NEVER fabricate titles, authors, PMIDs, or DOIs to reach 5 papers.
        - If `status == "error"`: Report that Europe PMC is temporarily unavailable and do NOT invent citations.

    Args:
        disease_name: Canonical disease name to search in literature and preprint servers.
        max_results: Maximum number of publications to return (strictly capped at 5; returns fewer if fewer exist).

    Returns:
        Strict `RecentPublicationsOutput` dictionary with `status`, `disease_name`,
        `returned_count` (<= 5), `publications` list, and `recovery_instructions`.
    """
    raw_disease = redact_pii((disease_name or "").strip())
    try:
        validated_in = FetchRecentPublicationsInput(
            disease_name=raw_disease,
            max_results=max_results,
        )
        cleaned_disease = validated_in.disease_name
        capped_max = validated_in.max_results
    except ValidationError as val_exc:
        return validate_and_dump(  # type: ignore[return-value]
            RecentPublicationsOutput,
            {
                "status": "validation_error",
                "disease_name": raw_disease,
                "returned_count": 0,
                "publications": [],
                "message": f"Invalid parameters for fetch_recent_publications: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "Pass a non-empty canonical `disease_name` (at least 2 characters) and "
                    "an integer `max_results` between 1 and 5."
                ),
            },
        )

    parent_span = trace.get_current_span()
    if parent_span and parent_span.is_recording():
        parent_span.set_attribute("literature.disease_name", cleaned_disease)

    med_query = f'TITLE_ABS:"{cleaned_disease}" AND SRC:MED'
    ppr_query = (
        f'TITLE_ABS:"{cleaned_disease}" AND SRC:PPR AND '
        f'(PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv" '
        f'OR PUBLISHER:"Cold Spring Harbor Laboratory")'
    )
    fallback_query = f'"{cleaned_disease}" AND (SRC:MED OR SRC:PPR)'

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            with tracer.start_as_current_span("ebi.europepmc.search") as epmc_span:
                epmc_span.set_attribute("literature.disease_name", cleaned_disease)

                async def _run_epmc_query(
                    query_str: str,
                ) -> tuple[int, list[dict[str, Any]]]:
                    resp = await _async_request_with_retry(
                        client,
                        "GET",
                        EUROPE_PMC_SEARCH_URL,
                        params={
                            "query": query_str,
                            "format": "json",
                            "resultType": "core",
                            "pageSize": capped_max * 2,
                            "sort": "FIRST_PDATE_D desc",
                        },
                    )
                    epmc_span.set_attribute("http.status_code", resp.status_code)
                    resp.raise_for_status()
                    payload = resp.json()
                    hits = int(payload.get("hitCount") or 0)
                    items = (payload.get("resultList") or {}).get("result") or []
                    return hits, items

                (med_hits, med_items), (ppr_hits, ppr_items) = await asyncio.gather(
                    _run_epmc_query(med_query),
                    _run_epmc_query(ppr_query),
                )
                total_hit_count = med_hits + ppr_hits
                combined_raw = list(med_items) + list(ppr_items)

                if not combined_raw:
                    fb_hits, fb_items = await _run_epmc_query(fallback_query)
                    total_hit_count = fb_hits
                    combined_raw = list(fb_items)

                epmc_span.set_attribute("literature.total_hit_count", total_hit_count)

            peer_reviewed_pool: list[dict[str, Any]] = []
            preprint_pool: list[dict[str, Any]] = []
            seen_keys: set[str] = set()

            for item in combined_raw:
                if (
                    item.get("source") or ""
                ).upper() == "PPR" and not _is_allowed_preprint_item(item):
                    continue

                formatted = _format_europe_pmc_record(item)
                dedup_key = (
                    formatted.get("doi")
                    or formatted.get("pmid")
                    or formatted.get("title")
                    or ""
                )
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                if formatted.get("is_preprint"):
                    preprint_pool.append(formatted)
                else:
                    peer_reviewed_pool.append(formatted)

            target_preprints = min(2, capped_max // 2) if capped_max >= 2 else 0
            target_peer = capped_max - target_preprints

            selected_peer = peer_reviewed_pool[:target_peer]
            selected_preprints = preprint_pool[:target_preprints]
            publications: list[dict[str, Any]] = []

            max_len = max(len(selected_peer), len(selected_preprints))
            for idx in range(max_len):
                if idx < len(selected_peer):
                    publications.append(selected_peer[idx])
                if (
                    idx < len(selected_preprints)
                    and len(publications) < capped_max
                ):
                    publications.append(selected_preprints[idx])

            for extra in (
                peer_reviewed_pool[target_peer:]
                + preprint_pool[target_preprints:]
            ):
                if len(publications) >= capped_max:
                    break
                publications.append(extra)

            publications = publications[:capped_max]
            preprint_count = sum(1 for p in publications if p.get("is_preprint"))
            if parent_span and parent_span.is_recording():
                parent_span.set_attribute("literature.total_hit_count", total_hit_count)
                parent_span.set_attribute(
                    "literature.returned_count", len(publications)
                )
                parent_span.set_attribute("literature.preprint_count", preprint_count)

            sparse_recovery: str | None = None
            if len(publications) == 0:
                sparse_recovery = (
                    f"Europe PMC returned 0 publications for '{cleaned_disease}'. "
                    "State explicitly that no indexed recent publications or bioRxiv/medRxiv preprints "
                    "were found—NEVER invent citations, PMIDs, or DOIs."
                )
            elif len(publications) < capped_max:
                sparse_recovery = (
                    f"Only {len(publications)} relevant publications exist in Europe PMC for '{cleaned_disease}'. "
                    f"Present ONLY these {len(publications)} papers and never invent additional references to reach {capped_max}."
                )

            return validate_and_dump(  # type: ignore[return-value]
                RecentPublicationsOutput,
                {
                    "status": "success",
                    "disease_name": cleaned_disease,
                    "max_allowed": MAX_PUBLICATIONS_CAP,
                    "returned_count": len(publications),
                    "includes_preprints_from": ["bioRxiv", "medRxiv"],
                    "publications": publications,
                    "recovery_instructions": sparse_recovery,
                },
            )
    except Exception as exc:
        logger.exception(
            "Error fetching recent publications for '%s'", cleaned_disease
        )
        return validate_and_dump(  # type: ignore[return-value]
            RecentPublicationsOutput,
            {
                "status": "error",
                "disease_name": cleaned_disease,
                "returned_count": 0,
                "publications": [],
                "message": f"Failed to fetch recent publications: {exc}",
                "recovery_instructions": (
                    "Europe PMC REST API experienced a transient error after retries. "
                    "Do NOT fabricate publications, PMIDs, or DOIs; report the temporary "
                    "literature service error in the dossier."
                ),
            },
        )


async def save_dossier_artifact(
    efo_id: Annotated[
        str,
        Field(
            min_length=2,
            max_length=100,
            description="The canonical EFO/MONDO identifier for the characterized disease.",
        ),
    ],
    markdown_content: Annotated[
        str,
        Field(
            min_length=5,
            description="Full Markdown text of the Disease Characterisation Dossier.",
        ),
    ],
    tool_context: ToolContext,
) -> SaveDossierArtifactOutput:
    """Saves the synthesized Disease Characterisation Dossier as a downloadable Markdown artifact.

    Recovery Instructions:
        - If `status == "validation_error"`: Provide a non-empty `efo_id` and the complete `markdown_content` string (>= 5 chars).
        - If `status == "saved_in_session_state"`: The external `ArtifactService` was not initialized, so the
          dossier was persisted in `session.state['latest_dossier_markdown']` and `characterized_diseases_index`.
          Do NOT retry `save_dossier_artifact`; simply output the complete Markdown dossier directly to the user.

    Args:
        efo_id: The canonical EFO/MONDO identifier for the characterized disease.
        markdown_content: Full Markdown text of the Disease Characterisation Dossier.

    Returns:
        Strict `SaveDossierArtifactOutput` dictionary with `status`, `artifact_filename`,
        `version`, and `recovery_instructions`.
    """
    raw_efo = redact_pii((efo_id or "").strip())
    raw_md = redact_pii(markdown_content or "")
    try:
        validated_in = SaveDossierArtifactInput(
            efo_id=raw_efo,
            markdown_content=raw_md,
        )
    except ValidationError as val_exc:
        return validate_and_dump(  # type: ignore[return-value]
            SaveDossierArtifactOutput,
            {
                "status": "validation_error",
                "artifact_filename": "",
                "version": 0,
                "message": f"Invalid parameters for save_dossier_artifact: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "Call `save_dossier_artifact` with a non-empty `efo_id` (e.g., 'EFO_0000768') "
                    "and the complete synthesized `markdown_content` string."
                ),
            },
        )

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", validated_in.efo_id)
    filename = f"{safe_id}_dossier.md"
    scrubbed_markdown = validated_in.markdown_content
    content_bytes = scrubbed_markdown.encode("utf-8")

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("disease.efo_id", safe_id)
        span.set_attribute("dossier.artifact_filename", filename)
        span.set_attribute("dossier.byte_size", len(content_bytes))

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["latest_dossier_filename"] = filename
        tool_context.state["latest_dossier_markdown"] = scrubbed_markdown

        existing_index = tool_context.state.get("characterized_diseases_index")
        index_dict: dict[str, dict[str, Any]] = (
            dict(existing_index) if isinstance(existing_index, dict) else {}
        )
        disease_name = str(
            tool_context.state.get("canonical_disease_name") or safe_id
        )
        icd10_str = str(
            tool_context.state.get("icd10_codes_str")
            or "ICD-10: Not mapped in ontology cross-references"
        )
        index_dict[safe_id] = {
            "efo_id": safe_id,
            "canonical_disease_name": disease_name,
            "icd10_codes_str": icd10_str,
            "artifact_filename": filename,
            "targets_report": str(tool_context.state.get("targets_report") or ""),
            "treatments_report": str(
                tool_context.state.get("treatments_report") or ""
            ),
            "literature_report": str(
                tool_context.state.get("literature_report") or ""
            ),
            "dossier_markdown": scrubbed_markdown,
        }
        tool_context.state["characterized_diseases_index"] = index_dict
        summary_blocks = [
            (
                f"### {entry['canonical_disease_name']} (`{entry['efo_id']}` | {entry['icd10_codes_str']})\n"
                f"- **Artifact**: `{entry['artifact_filename']}`\n"
                f"- **Dossier**:\n{entry['dossier_markdown']}"
            )
            for entry in index_dict.values()
        ]
        tool_context.state["characterized_diseases_summary"] = "\n\n---\n\n".join(
            summary_blocks
        )

        # Schedule non-blocking async background memory consolidation into the persistent vector store
        try:
            from app.memory import memory_consolidator

            state_obj = tool_context.state
            state_snapshot = (
                state_obj.to_dict()
                if hasattr(state_obj, "to_dict")
                else dict(state_obj)
            )
            inv_ctx = getattr(tool_context, "_invocation_context", None)
            session = getattr(inv_ctx, "session", None)
            memory_consolidator.schedule_consolidation(
                state_snapshot=state_snapshot,
                app_name=getattr(session, "app_name", None) or "app",
                user_id=getattr(session, "user_id", None) or "default_user",
                session_id=getattr(session, "id", None),
                memory_service=getattr(inv_ctx, "memory_service", None),
            )
        except Exception as bg_exc:
            logger.debug("Background memory consolidation schedule skipped: %s", bg_exc)

    try:
        part = types.Part(
            inline_data=types.Blob(
                mime_type="text/markdown",
                data=content_bytes,
            )
        )
        version = int(await tool_context.save_artifact(filename, part) or 0)
        return validate_and_dump(  # type: ignore[return-value]
            SaveDossierArtifactOutput,
            {
                "status": "success",
                "artifact_filename": filename,
                "version": version,
                "recovery_instructions": None,
            },
        )
    except Exception as exc:
        logger.debug("Artifact service save fallback used: %s", exc)
        return validate_and_dump(  # type: ignore[return-value]
            SaveDossierArtifactOutput,
            {
                "status": "saved_in_session_state",
                "artifact_filename": filename,
                "version": 0,
                "message": "ArtifactService not configured; dossier saved in session state.",
                "recovery_instructions": (
                    "Dossier is already persisted in `session.state['latest_dossier_markdown']` "
                    "and `characterized_diseases_index`. Do NOT retry `save_dossier_artifact`; "
                    "present the complete Markdown dossier directly to the user."
                ),
            },
        )
