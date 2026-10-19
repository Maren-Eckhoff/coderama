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

"""Unit tests for biomedical API tool contracts and ADK Workflow orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent import (
    app,
    check_ontology_resolution,
    disease_characterisation_pipeline,
    research_join,
    root_agent,
    unresolved_disease_handler,
)
from app.tools import (
    _extract_icd10_codes,
    fetch_known_targets,
    fetch_recent_publications,
    fetch_treatments_and_trials,
    resolve_disease_ontology,
    save_dossier_artifact,
)


class _DummyToolContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.route: str | None = None

    async def save_artifact(self, filename: str, part: Any) -> int:
        return 1


def test_app_and_orchestration_configuration() -> None:
    """Verify ADK App name, 5-event compaction config, and Workflow DAG topology."""
    from google.adk.workflow import JoinNode, Workflow

    assert app.name == "app"
    assert root_agent.name == "disease_characterisation"
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.compaction_interval == 5
    assert app.events_compaction_config.overlap_size == 1

    assert isinstance(disease_characterisation_pipeline, Workflow)
    assert isinstance(research_join, JoinNode)
    assert disease_characterisation_pipeline.graph is not None
    node_names = {node.name for node in disease_characterisation_pipeline.graph.nodes}
    assert {
        "disease_resolver_agent",
        "check_ontology_resolution",
        "unresolved_disease_handler",
        "targets_agent",
        "treatments_agent",
        "literature_agent",
        "research_join",
        "dossier_synthesizer_agent",
    }.issubset(node_names)


@pytest.mark.asyncio
async def test_short_circuit_routing_on_ontology_resolution() -> None:
    """Verify check_ontology_resolution routes 'resolved' vs 'unresolved' based on session state."""
    ctx_ok = _DummyToolContext()
    ctx_ok.state["resolution_status"] = "success"
    ctx_ok.state["efo_id"] = "EFO_0000768"
    out_ok = await check_ontology_resolution(ctx_ok, "Resolved IPF")  # type: ignore[arg-type]
    assert ctx_ok.route == "resolved"
    assert out_ok == "Resolved IPF"

    ctx_fail = _DummyToolContext()
    ctx_fail.state["resolution_status"] = "not_found"
    ctx_fail.state["efo_id"] = "Not yet resolved."
    out_fail = await check_ontology_resolution(ctx_fail, "No match")  # type: ignore[arg-type]
    assert ctx_fail.route == "unresolved"
    assert out_fail == "No match"

    short_circuit_msg = await unresolved_disease_handler(ctx_fail, "No match for 'XYZ'")  # type: ignore[arg-type]
    assert "Disease Ontology Resolution Unsuccessful" in short_circuit_msg
    assert "Disclaimer:" in short_circuit_msg


def test_extract_icd10_codes() -> None:
    """Verify ICD-10 extraction and deduplication from dbXRefs."""
    raw_xrefs = [
        "MONDO:0008345",
        "ICD10:J84.1",
        "ICD10CM:J84.112",
        "icd10:J84.1",
        "MeSH:D054990",
    ]
    codes = _extract_icd10_codes(raw_xrefs)
    assert codes == ["ICD10:J84.1", "ICD10:J84.112"]


@pytest.mark.asyncio
@patch("app.tools.httpx.AsyncClient")
async def test_resolve_disease_ontology_contract(mock_client_cls: MagicMock) -> None:
    """Verify resolve_disease_ontology returns EFO ID, ICD-10 codes, and populates structured state keys."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    search_response = MagicMock()
    search_response.status_code = 200
    search_response.json.return_value = {
        "data": {
            "search": {
                "hits": [
                    {
                        "id": "EFO_0000768",
                        "name": "idiopathic pulmonary fibrosis",
                        "description": "Chronic progressive fibrosing interstitial pneumonia.",
                    }
                ]
            }
        }
    }

    details_response = MagicMock()
    details_response.status_code = 200
    details_response.json.return_value = {
        "data": {
            "disease": {
                "id": "EFO_0000768",
                "name": "idiopathic pulmonary fibrosis",
                "description": "Chronic progressive fibrosing interstitial pneumonia.",
                "dbXRefs": ["ICD10:J84.1", "MONDO:0008345"],
                "synonyms": [
                    {
                        "relation": "exact",
                        "terms": ["IPF", "cryptogenic fibrosing alveolitis"],
                    }
                ],
                "therapeuticAreas": [
                    {"id": "OTAR_0000010", "name": "respiratory or thoracic disease"}
                ],
            }
        }
    }

    mock_client.post.side_effect = [search_response, details_response]

    ctx = _DummyToolContext()
    result = await resolve_disease_ontology("Idiopathic Pulmonary Fibrosis", ctx)  # type: ignore[arg-type]

    assert result["status"] == "success"
    assert result["efo_id"] == "EFO_0000768"
    assert result["icd10_codes"] == ["ICD10:J84.1"]
    assert "IPF" in result["synonyms"]
    assert ctx.state["resolution_status"] == "success"
    assert ctx.state["efo_id"] == "EFO_0000768"
    assert ctx.state["canonical_disease_name"] == "idiopathic pulmonary fibrosis"
    assert ctx.state["icd10_codes_str"] == "ICD10:J84.1"
    assert ctx.state["disease_context"]["efo_id"] == "EFO_0000768"


@pytest.mark.asyncio
@patch("app.tools.httpx.AsyncClient")
async def test_fetch_known_targets_contract(mock_client_cls: MagicMock) -> None:
    """Verify fetch_known_targets parses target symbols, scores, and tractability."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "disease": {
                "id": "EFO_0000768",
                "name": "idiopathic pulmonary fibrosis",
                "associatedTargets": {
                    "count": 120,
                    "rows": [
                        {
                            "score": 0.84219,
                            "target": {
                                "id": "ENSG00000105329",
                                "approvedSymbol": "TGFB1",
                                "approvedName": "transforming growth factor beta 1",
                                "biotype": "protein_coding",
                                "tractability": [
                                    {
                                        "label": "Clinical Precedence",
                                        "modality": "AB",
                                        "value": True,
                                    },
                                    {
                                        "label": "Predicted Tractable",
                                        "modality": "SM",
                                        "value": False,
                                    },
                                ],
                            },
                            "datatypeScores": [
                                {"id": "genetic_association", "score": 0.75},
                                {"id": "known_drug", "score": 0.91},
                            ],
                        }
                    ],
                },
            }
        }
    }
    mock_client.post.return_value = mock_resp

    result = await fetch_known_targets("EFO_0000768", 5)
    assert result["status"] == "success"
    assert result["total_associated_targets"] == 120
    assert len(result["targets"]) == 1
    assert result["targets"][0]["gene_symbol"] == "TGFB1"
    assert result["targets"][0]["ensembl_id"] == "ENSG00000105329"
    assert result["targets"][0]["tractability_positive_modalities"] == [
        "AB: Clinical Precedence"
    ]


@pytest.mark.asyncio
@patch("app.tools.httpx.AsyncClient")
async def test_fetch_treatments_and_trials_contract(mock_client_cls: MagicMock) -> None:
    """Verify fetch_treatments_and_trials concurrently fetches Open Targets drugs and ClinicalTrials.gov studies."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    ot_resp = MagicMock()
    ot_resp.status_code = 200
    ot_resp.json.return_value = {
        "data": {
            "disease": {
                "id": "EFO_0000768",
                "name": "idiopathic pulmonary fibrosis",
                "knownDrugs": {
                    "uniqueDrugs": 2,
                    "uniqueTargets": 2,
                    "count": 2,
                    "rows": [
                        {
                            "phase": 4,
                            "status": "Completed",
                            "drugId": "CHEMBL2103830",
                            "prefName": "NINTEDANIB",
                            "drugType": "Small molecule",
                            "mechanismOfAction": "Fibroblast growth factor receptor inhibitor",
                            "targetId": "ENSG00000077782",
                            "target": {
                                "approvedSymbol": "FGFR1",
                                "approvedName": "fibroblast growth factor receptor 1",
                            },
                            "ctIds": ["NCT01335464"],
                        }
                    ],
                },
            }
        }
    }

    ct_resp = MagicMock()
    ct_resp.status_code = 200
    ct_resp.json.return_value = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT01335464",
                        "briefTitle": "Safety and Efficacy of BIBF 1120 in IPF",
                    },
                    "statusModule": {"overallStatus": "COMPLETED"},
                    "designModule": {"phases": ["PHASE3"]},
                    "armsInterventionsModule": {
                        "interventions": [{"type": "DRUG", "name": "Nintedanib"}]
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "Boehringer Ingelheim"}
                    },
                }
            }
        ]
    }

    mock_client.post.return_value = ot_resp
    mock_client.get.return_value = ct_resp

    result = await fetch_treatments_and_trials(
        "EFO_0000768", "Idiopathic Pulmonary Fibrosis", 5
    )
    assert result["status"] == "success"
    assert len(result["known_drugs"]) == 1
    assert result["known_drugs"][0]["chembl_id"] == "CHEMBL2103830"
    assert len(result["clinical_trials"]) == 1
    assert result["clinical_trials"][0]["nct_id"] == "NCT01335464"


@pytest.mark.asyncio
@patch("app.tools.httpx.AsyncClient")
async def test_fetch_recent_publications_enforces_max_5_and_balanced_preprints(
    mock_client_cls: MagicMock,
) -> None:
    """Verify fetch_recent_publications caps output at <= 5 and balances peer-reviewed + bioRxiv/medRxiv preprints."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    med_items = [
        {
            "id": f"MED_{i}",
            "source": "MED",
            "title": f"Peer-Reviewed Study {i} on Fibrosis",
            "authorString": "Smith J et al.",
            "journalTitle": "Nature Medicine",
            "bookOrReportDetails": {"publisher": ""},
            "pubYear": "2026",
            "firstPublicationDate": f"2026-0{i + 1}-01",
            "doi": f"10.1038/s41591-026-000{i}",
            "pmid": f"3900000{i}",
            "citedByCount": i,
            "abstractText": "<p>Abstract text for peer-reviewed study.</p>",
        }
        for i in range(5)
    ]

    ppr_items = [
        {
            "id": f"PPR_{i}",
            "source": "PPR",
            "title": f"Preprint Study {i} on Fibrosis",
            "authorString": "Lee K et al.",
            "journalTitle": "bioRxiv" if i == 0 else "medRxiv",
            "bookOrReportDetails": {"publisher": "bioRxiv" if i == 0 else "medRxiv"},
            "pubYear": "2026",
            "firstPublicationDate": f"2026-0{i + 1}-02",
            "doi": f"10.1101/2026.01.0{i}",
            "pmid": None,
            "citedByCount": 0,
            "abstractText": "<p>Abstract text for preprint study.</p>",
        }
        for i in range(4)
    ]

    med_resp = MagicMock()
    med_resp.status_code = 200
    med_resp.json.return_value = {"hitCount": 5, "resultList": {"result": med_items}}

    ppr_resp = MagicMock()
    ppr_resp.status_code = 200
    ppr_resp.json.return_value = {"hitCount": 4, "resultList": {"result": ppr_items}}

    mock_client.get.side_effect = [med_resp, ppr_resp]

    # Even if caller passes max_results=10, tool must strictly cap at 5 (3 peer-reviewed + 2 preprints)
    result = await fetch_recent_publications("Idiopathic Pulmonary Fibrosis", 10)
    assert result["status"] == "success"
    assert result["returned_count"] == 5
    assert len(result["publications"]) == 5

    badges = [p["source_badge"] for p in result["publications"]]
    assert badges.count("[Peer-Reviewed]") == 3
    assert "[Preprint - bioRxiv]" in badges
    assert "[Preprint - medRxiv]" in badges


@pytest.mark.asyncio
async def test_save_dossier_artifact() -> None:
    """Verify save_dossier_artifact writes artifact and updates session state."""
    ctx = _DummyToolContext()
    res = await save_dossier_artifact("EFO_0000768", "# Dossier Content", ctx)  # type: ignore[arg-type]
    assert res["status"] == "success"
    assert res["artifact_filename"] == "EFO_0000768_dossier.md"
    assert ctx.state["latest_dossier_filename"] == "EFO_0000768_dossier.md"


def test_pii_redaction_preserves_biomedical_ids() -> None:
    """Verify redact_pii scrubs PII/PHI while preserving EFO, ICD-10, ENSG, ChEMBL, NCT, PMID, and DOIs."""
    from app.app_utils.telemetry import redact_pii

    raw_text = (
        "Patient jane.doe@example.org (SSN 123-45-6789, Phone 415-555-0199, "
        "MRN: 9948210, DOB: 04/12/1965) diagnosed with EFO_0000768 (ICD10:J84.1), "
        "target ENSG00000140694, drug CHEMBL1256391, trial NCT05571059, "
        "paper PMID: 42827411 and DOI 10.1101/2026.10.07.757436."
    )
    scrubbed = redact_pii(raw_text)

    assert "jane.doe@example.org" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "415-555-0199" not in scrubbed
    assert "9948210" not in scrubbed
    assert "04/12/1965" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_MRN]" in scrubbed
    assert "[REDACTED_DOB]" in scrubbed

    # Biomedical identifiers must remain untouched
    assert "EFO_0000768" in scrubbed
    assert "ICD10:J84.1" in scrubbed
    assert "ENSG00000140694" in scrubbed
    assert "CHEMBL1256391" in scrubbed
    assert "NCT05571059" in scrubbed
    assert "42827411" in scrubbed
    assert "10.1101/2026.10.07.757436" in scrubbed


@pytest.mark.asyncio
async def test_observability_plugin_and_structured_formatter() -> None:
    """Verify DiseaseObservabilityPlugin scrubs user PII and logs structured JSON."""
    import json
    import logging

    from google.genai import types

    from app.app_utils.telemetry import (
        DiseaseObservabilityPlugin,
        TraceCorrelatedJsonFormatter,
    )

    plugin = DiseaseObservabilityPlugin()
    msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Characterize IPF for patient john@hospital.org (SSN 111-22-3333)"
            )
        ],
    )
    sanitized = await plugin.on_user_message_callback(
        invocation_context=MagicMock(), user_message=msg
    )
    assert sanitized is not None
    assert "john@hospital.org" not in sanitized.parts[0].text
    assert "[REDACTED_EMAIL]" in sanitized.parts[0].text
    assert "[REDACTED_SSN]" in sanitized.parts[0].text

    formatter = TraceCorrelatedJsonFormatter(project_id="kaggle-sandbox-686020")
    record = logging.LogRecord(
        name="disease_characterisation.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Query by user@example.com for EFO_0000768",
        args=(),
        exc_info=None,
    )
    record.structured_data = {"event_type": "test_event", "efo_id": "EFO_0000768"}
    formatted_json = json.loads(formatter.format(record))
    assert formatted_json["severity"] == "INFO"
    assert "[REDACTED_EMAIL]" in formatted_json["message"]
    assert formatted_json["efo_id"] == "EFO_0000768"


def test_required_tool_trajectory_metric() -> None:
    """Verify trajectory_metrics.evaluate scores full and partial tool coverage."""
    from tests.eval.trajectory_metrics import REQUIRED_TOOLS, evaluate

    events = [
        {
            "author": "sub_agent",
            "content": {
                "parts": [
                    {"function_call": {"name": tool_name, "args": {}}},
                    {
                        "function_response": {
                            "name": tool_name,
                            "response": {"status": "success"},
                        }
                    },
                ]
            },
        }
        for tool_name in REQUIRED_TOOLS
    ]
    instance = {"agent_data": {"turns": [{"events": events}]}}
    verdict = evaluate(instance)
    assert verdict["score"] == 1.0
    assert "5/5" in verdict["explanation"]
