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

import logging
import re
from typing import Any

import httpx
from google.adk.tools import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
EBI_OLS4_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_PUBLICATIONS_CAP = 5


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


def _lookup_icd10_via_ols4(
    disease_name: str, efo_id: str, client: httpx.Client
) -> list[str]:
    """Fallback lookup of ICD-10 cross-references via the EBI OLS4 REST API."""
    icd10_codes: list[str] = []
    seen: set[str] = set()
    try:
        resp = client.get(
            EBI_OLS4_SEARCH_URL,
            params={
                "q": efo_id if efo_id else disease_name,
                "FieldList": "iri,label,short_form,obo_id,annotations_trimmed",
                "rows": 5,
            },
        )
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
    except Exception as exc:
        logger.debug("OLS4 ICD-10 fallback lookup failed: %s", exc)
    return icd10_codes


def resolve_disease_ontology(
    disease_query: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Resolves a free-text disease query into canonical EFO and ICD-10 ontology identifiers.

    Args:
        disease_query: The disease name, synonym, or acronym entered by the user (e.g., 'Idiopathic Pulmonary Fibrosis', 'NSCLC', 'Crohn disease').

    Returns:
        dict with 'status', 'efo_id', 'name', 'icd10_codes', 'description',
        'synonyms', 'therapeutic_areas', and 'open_targets_url'.
    """
    cleaned_query = (disease_query or "").strip()
    if not cleaned_query:
        return {
            "status": "error",
            "message": "Disease query cannot be empty.",
        }

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

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            search_resp = client.post(
                OPEN_TARGETS_GRAPHQL_URL,
                json={"query": search_gql, "variables": {"queryString": cleaned_query}},
            )
            search_resp.raise_for_status()
            search_data = search_resp.json().get("data", {}) or {}
            hits = (search_data.get("search") or {}).get("hits") or []

            if not hits:
                return {
                    "status": "not_found",
                    "query": cleaned_query,
                    "message": f"No canonical disease entry found in Open Targets for '{cleaned_query}'.",
                }

            top_hit = hits[0]
            efo_id = top_hit["id"]

            details_resp = client.post(
                OPEN_TARGETS_GRAPHQL_URL,
                json={"query": details_gql, "variables": {"efoId": efo_id}},
            )
            details_resp.raise_for_status()
            disease_obj = (details_resp.json().get("data") or {}).get("disease") or {}

            db_xrefs = disease_obj.get("dbXRefs") or []
            icd10_codes = _extract_icd10_codes(db_xrefs)
            if not icd10_codes:
                icd10_codes = _lookup_icd10_via_ols4(
                    disease_obj.get("name") or top_hit.get("name") or cleaned_query,
                    efo_id,
                    client,
                )

            synonyms: list[str] = []
            for group in disease_obj.get("synonyms") or []:
                for term in group.get("terms") or []:
                    if term and term not in synonyms:
                        synonyms.append(term)

            therapeutic_areas = [
                {"id": ta.get("id"), "name": ta.get("name")}
                for ta in (disease_obj.get("therapeuticAreas") or [])
                if ta.get("name")
            ]

            alternative_matches = [
                {"efo_id": h.get("id"), "name": h.get("name")}
                for h in hits[1:4]
                if h.get("id")
            ]

            result = {
                "status": "success",
                "query": cleaned_query,
                "efo_id": efo_id,
                "name": disease_obj.get("name") or top_hit.get("name"),
                "icd10_codes": icd10_codes,
                "other_xrefs": [
                    ref
                    for ref in db_xrefs[:15]
                    if isinstance(ref, str)
                    and not ref.upper().startswith(("ICD10:", "ICD10CM:"))
                ],
                "description": disease_obj.get("description")
                or top_hit.get("description")
                or "",
                "synonyms": synonyms[:10],
                "therapeutic_areas": therapeutic_areas,
                "alternative_matches": alternative_matches,
                "open_targets_url": f"https://platform.opentargets.org/disease/{efo_id}",
            }

            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["disease_context"] = result

            return result
    except Exception as exc:
        logger.exception("Error resolving disease ontology for '%s'", cleaned_query)
        return {
            "status": "error",
            "query": cleaned_query,
            "message": f"Failed to resolve disease ontology: {exc}",
        }


def fetch_known_targets(
    efo_id: str,
    limit: int,
) -> dict[str, Any]:
    """Fetches top-ranked known biological targets associated with a disease EFO/MONDO ID from Open Targets.

    Args:
        efo_id: Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').
        limit: Maximum number of top targets to return (capped between 1 and 15).

    Returns:
        dict with 'status', 'efo_id', 'total_associated_targets', and 'targets' list
        containing gene symbols, Ensembl IDs, association scores, and tractability.
    """
    cleaned_efo = (efo_id or "").strip()
    if not cleaned_efo:
        return {"status": "error", "message": "efo_id cannot be empty."}

    capped_limit = min(max(int(limit), 1), 15)

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
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            resp = client.post(
                OPEN_TARGETS_GRAPHQL_URL,
                json={
                    "query": targets_gql,
                    "variables": {"efoId": cleaned_efo, "size": capped_limit},
                },
            )
            resp.raise_for_status()
            disease_data = (resp.json().get("data") or {}).get("disease")
            if not disease_data:
                return {
                    "status": "not_found",
                    "efo_id": cleaned_efo,
                    "message": f"No disease found for EFO ID '{cleaned_efo}'.",
                }

            assoc = disease_data.get("associatedTargets") or {}
            rows = assoc.get("rows") or []
            targets_list: list[dict[str, Any]] = []

            for row in rows:
                target_info = row.get("target") or {}
                ensembl_id = target_info.get("id") or ""
                tractability_items = [
                    f"{t.get('modality')}: {t.get('label')}"
                    for t in (target_info.get("tractability") or [])
                    if t.get("value") is True
                ]
                datatype_scores = {
                    dt.get("id"): round(float(dt.get("score", 0.0)), 4)
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

            return {
                "status": "success",
                "efo_id": cleaned_efo,
                "disease_name": disease_data.get("name"),
                "total_associated_targets": assoc.get("count", len(targets_list)),
                "targets": targets_list,
            }
    except Exception as exc:
        logger.exception("Error fetching known targets for '%s'", cleaned_efo)
        return {
            "status": "error",
            "efo_id": cleaned_efo,
            "message": f"Failed to fetch known targets: {exc}",
        }


def fetch_treatments_and_trials(
    efo_id: str,
    disease_name: str,
    limit: int,
) -> dict[str, Any]:
    """Fetches existing approved/investigational drugs from Open Targets and clinical trials from ClinicalTrials.gov v2.

    Args:
        efo_id: Canonical ontology identifier (e.g., 'EFO_0000768').
        disease_name: Canonical disease name for querying ClinicalTrials.gov (e.g., 'Idiopathic Pulmonary Fibrosis').
        limit: Maximum number of drugs and clinical trials to return (capped between 1 and 15).

    Returns:
        dict with 'status', 'efo_id', 'known_drugs' (ChEMBL ID, drug name, modality, MoA, phase, target),
        and 'clinical_trials' (NCT ID, title, status, phases, interventions).
    """
    cleaned_efo = (efo_id or "").strip()
    cleaned_name = (disease_name or "").strip()
    capped_limit = min(max(int(limit), 1), 15)

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
            ctIds
          }
        }
      }
    }
    """

    known_drugs: list[dict[str, Any]] = []
    unique_drugs_count = 0
    clinical_trials: list[dict[str, Any]] = []
    errors: list[str] = []

    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        # 1. Fetch known drugs from Open Targets
        if cleaned_efo:
            try:
                resp = client.post(
                    OPEN_TARGETS_GRAPHQL_URL,
                    json={
                        "query": drugs_gql,
                        "variables": {"efoId": cleaned_efo, "size": capped_limit * 3},
                    },
                )
                resp.raise_for_status()
                disease_data = (resp.json().get("data") or {}).get("disease") or {}
                kd = disease_data.get("knownDrugs") or {}
                unique_drugs_count = kd.get("uniqueDrugs", 0)

                seen_drug_targets: set[tuple[str, str]] = set()
                for row in kd.get("rows") or []:
                    drug_id = row.get("drugId") or ""
                    target_id = row.get("targetId") or ""
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
                            "target_ensembl_id": target_id,
                            "clinical_trial_ids": (row.get("ctIds") or [])[:5],
                            "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{drug_id}/"
                            if drug_id
                            else None,
                        }
                    )
                    if len(known_drugs) >= capped_limit:
                        break
            except Exception as exc:
                logger.warning("Open Targets knownDrugs query failed: %s", exc)
                errors.append(f"Open Targets knownDrugs error: {exc}")

        # 2. Fetch active/recent studies from ClinicalTrials.gov v2
        if cleaned_name:
            try:
                ct_resp = client.get(
                    CLINICAL_TRIALS_API_URL,
                    params={
                        "query.cond": cleaned_name,
                        "pageSize": min(capped_limit, 10),
                        "format": "json",
                    },
                )
                ct_resp.raise_for_status()
                studies = ct_resp.json().get("studies") or []
                for study in studies[:capped_limit]:
                    proto = study.get("protocolSection") or {}
                    id_mod = proto.get("identificationModule") or {}
                    status_mod = proto.get("statusModule") or {}
                    design_mod = proto.get("designModule") or {}
                    arms_mod = proto.get("armsInterventionsModule") or {}
                    sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}

                    nct_id = id_mod.get("nctId") or ""
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
                            "phases": design_mod.get("phases") or [],
                            "interventions": interventions[:5],
                            "lead_sponsor": (
                                sponsor_mod.get("leadSponsor") or {}
                            ).get("name"),
                            "url": f"https://clinicaltrials.gov/study/{nct_id}"
                            if nct_id
                            else None,
                        }
                    )
            except Exception as exc:
                logger.warning("ClinicalTrials.gov API v2 query failed: %s", exc)
                errors.append(f"ClinicalTrials.gov error: {exc}")

    if not known_drugs and not clinical_trials and errors:
        return {
            "status": "error",
            "efo_id": cleaned_efo,
            "disease_name": cleaned_name,
            "message": "; ".join(errors),
        }

    return {
        "status": "success",
        "efo_id": cleaned_efo,
        "disease_name": cleaned_name,
        "unique_known_drugs_count": unique_drugs_count,
        "known_drugs": known_drugs,
        "clinical_trials": clinical_trials,
        "partial_warnings": errors,
    }


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
    is_preprint = source == "PPR" or "biorxiv" in combined_pub_info or "medrxiv" in combined_pub_info
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
        "venue": venue,
        "publication_year": item.get("pubYear"),
        "first_publication_date": item.get("firstPublicationDate"),
        "publication_type": publication_type,
        "source_badge": source_badge,
        "is_preprint": is_preprint,
        "pmid": pmid or None,
        "doi": doi or None,
        "cited_by_count": item.get("citedByCount", 0),
        "url": url,
        "abstract_snippet": abstract_clean,
    }


def fetch_recent_publications(
    disease_name: str,
    max_results: int,
) -> dict[str, Any]:
    """Fetches up to 5 recent peer-reviewed publications (PubMed) and preprints (bioRxiv/medRxiv) for a disease.

    Args:
        disease_name: Canonical disease name to search in literature and preprint servers.
        max_results: Maximum number of publications to return (strictly capped at 5; returns fewer if fewer exist).

    Returns:
        dict with 'status', 'disease_name', 'returned_count' (<= 5), and 'publications' list.
    """
    cleaned_disease = (disease_name or "").strip()
    if not cleaned_disease:
        return {"status": "error", "message": "disease_name cannot be empty."}

    capped_max = min(max(int(max_results), 1), MAX_PUBLICATIONS_CAP)

    # Query Europe PMC for both peer-reviewed (SRC:MED) and bioRxiv/medRxiv preprints (SRC:PPR)
    combined_query = (
        f'TITLE_ABS:"{cleaned_disease}" AND '
        f'(SRC:MED OR (SRC:PPR AND (PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv" '
        f'OR PUBLISHER:"Cold Spring Harbor Laboratory")))'
    )
    fallback_query = f'"{cleaned_disease}" AND (SRC:MED OR SRC:PPR)'

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
            resp = client.get(
                EUROPE_PMC_SEARCH_URL,
                params={
                    "query": combined_query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": capped_max * 2,
                    "sort": "FIRST_PDATE_D desc",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw_results = (data.get("resultList") or {}).get("result") or []

            if not raw_results:
                fallback_resp = client.get(
                    EUROPE_PMC_SEARCH_URL,
                    params={
                        "query": fallback_query,
                        "format": "json",
                        "resultType": "core",
                        "pageSize": capped_max * 2,
                        "sort": "FIRST_PDATE_D desc",
                    },
                )
                fallback_resp.raise_for_status()
                raw_results = (
                    (fallback_resp.json().get("resultList") or {}).get("result") or []
                )

            publications: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for item in raw_results:
                # Filter out non-bioRxiv/medRxiv preprints if source is PPR
                if (item.get("source") or "").upper() == "PPR":
                    pub_text = (
                        f"{item.get('journalTitle', '')} "
                        f"{(item.get('bookOrReportDetails') or {}).get('publisher', '')} "
                        f"{item.get('doi', '')}"
                    ).lower()
                    if not any(
                        tok in pub_text
                        for tok in ("biorxiv", "medrxiv", "cold spring harbor", "10.1101/")
                    ):
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
                publications.append(formatted)
                if len(publications) >= capped_max:
                    break

            return {
                "status": "success",
                "disease_name": cleaned_disease,
                "max_allowed": MAX_PUBLICATIONS_CAP,
                "returned_count": len(publications),
                "includes_preprints_from": ["bioRxiv", "medRxiv"],
                "publications": publications,
            }
    except Exception as exc:
        logger.exception("Error fetching recent publications for '%s'", cleaned_disease)
        return {
            "status": "error",
            "disease_name": cleaned_disease,
            "message": f"Failed to fetch recent publications: {exc}",
        }


async def save_dossier_artifact(
    efo_id: str,
    markdown_content: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Saves the synthesized Disease Characterisation Dossier as a downloadable Markdown artifact.

    Args:
        efo_id: The canonical EFO/MONDO identifier for the characterized disease.
        markdown_content: Full Markdown text of the Disease Characterisation Dossier.

    Returns:
        dict with 'status', 'artifact_filename', and 'version'.
    """
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", (efo_id or "disease").strip())
    filename = f"{safe_id}_dossier.md"
    content_bytes = (markdown_content or "").encode("utf-8")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["latest_dossier_filename"] = filename
        tool_context.state["latest_dossier_markdown"] = markdown_content

    try:
        part = types.Part(
            inline_data=types.Blob(
                mime_type="text/markdown",
                data=content_bytes,
            )
        )
        version = await tool_context.save_artifact(filename, part)
        return {
            "status": "success",
            "artifact_filename": filename,
            "version": version,
        }
    except Exception as exc:
        logger.debug("Artifact service save fallback used: %s", exc)
        return {
            "status": "saved_in_session_state",
            "artifact_filename": filename,
            "version": 0,
        }
