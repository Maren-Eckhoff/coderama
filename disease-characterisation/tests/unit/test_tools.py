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
    ambiguous_disease_handler,
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
        self.function_call_id: str | None = "call-123"
        self.tool_confirmation: Any = None
        self.confirmation_requests: list[dict[str, Any]] = []

    def request_confirmation(
        self, *, hint: str | None = None, payload: Any | None = None
    ) -> None:
        self.confirmation_requests.append({"hint": hint, "payload": payload})

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
        "ambiguous_disease_handler",
        "unresolved_disease_handler",
        "targets_agent",
        "treatments_agent",
        "literature_agent",
        "research_join",
        "dossier_synthesizer_agent",
    }.issubset(node_names)


@pytest.mark.asyncio
async def test_short_circuit_and_ambiguous_routing_on_ontology_resolution() -> None:
    """Verify check_ontology_resolution routes 'resolved', 'ambiguous', and 'unresolved' based on session state."""
    ctx_ok = _DummyToolContext()
    ctx_ok.state["resolution_status"] = "success"
    ctx_ok.state["efo_id"] = "EFO_0000768"
    out_ok = await check_ontology_resolution(ctx_ok, "Resolved IPF")  # type: ignore[arg-type]
    assert ctx_ok.route == "resolved"
    assert out_ok == "Resolved IPF"

    ctx_amb = _DummyToolContext()
    ctx_amb.state["resolution_status"] = "ambiguous"
    ctx_amb.state["ambiguous_candidates"] = [
        {"name": "type 2 diabetes mellitus", "efo_id": "MONDO_0005148", "description": "Adult-onset diabetes."},
        {"name": "type 1 diabetes mellitus", "efo_id": "MONDO_0005147", "description": "Autoimmune diabetes."},
    ]
    out_amb = await check_ontology_resolution(ctx_amb, "Ambiguous diabetes")  # type: ignore[arg-type]
    assert ctx_amb.route == "ambiguous"
    assert out_amb == "Ambiguous diabetes"

    amb_msg = await ambiguous_disease_handler(ctx_amb, "Multiple matches found.")  # type: ignore[arg-type]
    assert "Human-in-the-Loop Disambiguation Required" in amb_msg
    assert "MONDO_0005148" in amb_msg
    assert "MONDO_0005147" in amb_msg

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
    assert result["resolution_confidence"] == "exact_name"
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
async def test_resolve_disease_ontology_direct_id_and_ambiguous_disambiguation(
    mock_client_cls: MagicMock,
) -> None:
    """Verify direct EFO ID lookup and HITL disambiguation on ambiguous multi-hit queries."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client

    # 1. Direct ID lookup (skips SearchDisease and calls DiseaseDetails directly)
    details_response = MagicMock()
    details_response.status_code = 200
    details_response.json.return_value = {
        "data": {
            "disease": {
                "id": "EFO_0000768",
                "name": "idiopathic pulmonary fibrosis",
                "description": "Chronic progressive fibrosing interstitial pneumonia.",
                "dbXRefs": ["ICD10:J84.1"],
                "synonyms": [{"relation": "exact", "terms": ["IPF"]}],
                "therapeuticAreas": [],
            }
        }
    }
    mock_client.post.side_effect = [details_response]
    ctx_direct = _DummyToolContext()
    direct_res = await resolve_disease_ontology("EFO:0000768", ctx_direct)  # type: ignore[arg-type]
    assert direct_res["status"] == "success"
    assert direct_res["resolution_confidence"] == "direct_id"
    assert direct_res["efo_id"] == "EFO_0000768"

    # 2. Ambiguous query ("muscular dystrophy" matching Duchenne, Becker, and Myotonic without exact match)
    amb_search_resp = MagicMock()
    amb_search_resp.status_code = 200
    amb_search_resp.json.return_value = {
        "data": {
            "search": {
                "hits": [
                    {
                        "id": "MONDO_0010679",
                        "name": "Duchenne muscular dystrophy",
                        "description": "Severe X-linked muscular dystrophy.",
                    },
                    {
                        "id": "MONDO_0010311",
                        "name": "Becker muscular dystrophy",
                        "description": "X-linked recessive muscular dystrophy.",
                    },
                ]
            }
        }
    }
    amb_details_resp = MagicMock()
    amb_details_resp.status_code = 200
    amb_details_resp.json.return_value = {
        "data": {
            "disease": {
                "id": "MONDO_0010679",
                "name": "Duchenne muscular dystrophy",
                "description": "Severe X-linked muscular dystrophy.",
                "dbXRefs": ["ICD10:G71.0"],
                "synonyms": [{"relation": "exact", "terms": ["DMD"]}],
                "therapeuticAreas": [],
            }
        }
    }
    mock_client.post.side_effect = [amb_search_resp, amb_details_resp]
    ctx_amb = _DummyToolContext()
    amb_res = await resolve_disease_ontology("dystrophy", ctx_amb)  # type: ignore[arg-type]

    assert amb_res["status"] == "ambiguous"
    assert amb_res["requires_human_disambiguation"] is True
    assert amb_res["resolution_confidence"] == "ambiguous"
    assert len(amb_res["candidates"]) == 2
    assert ctx_amb.state["resolution_status"] == "ambiguous"
    assert len(ctx_amb.confirmation_requests) == 1
    assert (
        ctx_amb.confirmation_requests[0]["payload"]["default_efo_id"]
        == "MONDO_0010679"
    )


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
    """Verify save_dossier_artifact writes artifact and consolidates multiple diseases in session state."""
    ctx = _DummyToolContext()
    ctx.state["canonical_disease_name"] = "idiopathic pulmonary fibrosis"
    ctx.state["icd10_codes_str"] = "ICD10:J84.1"
    res1 = await save_dossier_artifact("EFO_0000768", "# IPF Dossier Content", ctx)  # type: ignore[arg-type]
    assert res1["status"] == "success"
    assert res1["artifact_filename"] == "EFO_0000768_dossier.md"
    assert ctx.state["latest_dossier_filename"] == "EFO_0000768_dossier.md"

    ctx.state["canonical_disease_name"] = "Crohn's disease"
    ctx.state["icd10_codes_str"] = "ICD10:K50"
    res2 = await save_dossier_artifact("EFO_0000384", "# Crohn Dossier Content", ctx)  # type: ignore[arg-type]
    assert res2["status"] == "success"
    assert res2["artifact_filename"] == "EFO_0000384_dossier.md"

    # Both diseases must remain preserved in characterized_diseases_index & summary
    assert "EFO_0000768" in ctx.state["characterized_diseases_index"]
    assert "EFO_0000384" in ctx.state["characterized_diseases_index"]
    assert "# IPF Dossier Content" in ctx.state["characterized_diseases_summary"]
    assert "# Crohn Dossier Content" in ctx.state["characterized_diseases_summary"]


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


@pytest.mark.asyncio
async def test_strict_input_output_json_schemas_and_recovery_instructions() -> None:
    """Verify strict Pydantic input/output JSON schemas, ADK FunctionTool declarations, and explicit recovery_instructions."""
    from google.adk.tools.function_tool import FunctionTool
    from pydantic import ValidationError

    from app.schemas import (
        DiseaseOntologyOutput,
        KnownTargetsOutput,
        RecentPublicationsOutput,
        SaveDossierArtifactOutput,
        TreatmentsAndTrialsOutput,
        validate_and_dump,
    )

    # 1. Verify ADK FunctionTool generates strict parameters_json_schema and response_json_schema
    for fn, out_model in [
        (resolve_disease_ontology, DiseaseOntologyOutput),
        (fetch_known_targets, KnownTargetsOutput),
        (fetch_treatments_and_trials, TreatmentsAndTrialsOutput),
        (fetch_recent_publications, RecentPublicationsOutput),
        (save_dossier_artifact, SaveDossierArtifactOutput),
    ]:
        ft = FunctionTool(fn)
        decl = ft._get_declaration()
        assert decl is not None
        assert decl.parameters_json_schema is not None
        assert decl.response_json_schema is not None
        # Strict output JSON schema must forbid extra properties
        assert decl.response_json_schema.get("additionalProperties") is False
        assert "recovery_instructions" in decl.response_json_schema.get(
            "properties", {}
        )
        # Model-level JSON schema check
        json_schema = out_model.model_json_schema()
        assert json_schema.get("additionalProperties") is False

    # 2. Verify extra="forbid" rejects undeclared fields in output validation
    with pytest.raises(ValidationError):
        validate_and_dump(
            KnownTargetsOutput,
            {
                "status": "success",
                "efo_id": "EFO_0000768",
                "unexpected_extra_field": "forbidden",
            },
        )

    # 3. Verify invalid inputs trigger status="validation_error" with actionable recovery_instructions
    ctx = _DummyToolContext()
    bad_ontology = await resolve_disease_ontology("x", ctx)  # type: ignore[arg-type]
    assert bad_ontology["status"] == "validation_error"
    assert bad_ontology["recovery_instructions"] is not None
    assert "resolve_disease_ontology" in bad_ontology["recovery_instructions"]

    bad_targets = await fetch_known_targets("INVALID-ID-FORMAT", 10)
    assert bad_targets["status"] == "validation_error"
    assert bad_targets["recovery_instructions"] is not None
    assert "EFO_0000768" in bad_targets["recovery_instructions"]

    bad_treatments = await fetch_treatments_and_trials("INVALID_ID", "IPF", 10)
    assert bad_treatments["status"] == "validation_error"
    assert bad_treatments["recovery_instructions"] is not None
    assert "EFO_0000768" in bad_treatments["recovery_instructions"]

    bad_pubs = await fetch_recent_publications("   ", 5)
    assert bad_pubs["status"] == "validation_error"
    assert bad_pubs["recovery_instructions"] is not None
    assert "disease_name" in bad_pubs["recovery_instructions"]

    bad_artifact = await save_dossier_artifact("EFO_0000768", "too short", ctx)  # type: ignore[arg-type]
    assert bad_artifact["status"] == "validation_error"
    assert bad_artifact["recovery_instructions"] is not None
    assert "save_dossier_artifact" in bad_artifact["recovery_instructions"]


@pytest.mark.asyncio
async def test_persistent_vector_memory_and_async_background_consolidation(
    tmp_path: Any,
) -> None:
    """Verify persistent SQLite vector store, cosine similarity search, and non-blocking async background memory consolidation."""
    from google.adk.tools import load_memory, preload_memory

    from app.app_utils import services
    from app.memory import (
        AsyncMemoryConsolidator,
        PersistentVectorMemoryService,
        compute_text_embedding,
        cosine_similarity,
        search_long_term_disease_memory,
        trigger_async_memory_consolidation,
    )

    # 1. Verify root_agent has preload_memory, load_memory, search_long_term_disease_memory, and after_agent_callback
    assert preload_memory in root_agent.tools
    assert load_memory in root_agent.tools
    assert search_long_term_disease_memory in root_agent.tools
    assert root_agent.after_agent_callback == trigger_async_memory_consolidation
    assert isinstance(
        services.get_memory_service(), PersistentVectorMemoryService
    )

    # 2. Verify embedding cosine similarity ranks semantically related biomedical text higher
    vec_ipf = compute_text_embedding(
        "EFO_0000768 idiopathic pulmonary fibrosis TGFB1 pirfenidone nintedanib lung"
    )
    vec_ipf_query = compute_text_embedding(
        "pulmonary fibrosis TGFB1 nintedanib EFO_0000768"
    )
    vec_unrelated = compute_text_embedding(
        "MONDO_0005148 type 2 diabetes mellitus insulin metformin"
    )
    assert cosine_similarity(vec_ipf, vec_ipf_query) > cosine_similarity(
        vec_ipf, vec_unrelated
    )

    # 3. Verify AsyncMemoryConsolidator spawns asyncio.create_task and persists into PersistentVectorMemoryService
    db_file = tmp_path / "test_vector_memory.db"
    vec_service = PersistentVectorMemoryService(db_path=db_file)
    consolidator = AsyncMemoryConsolidator()

    state_snapshot = {
        "resolution_status": "success",
        "efo_id": "EFO_0000768",
        "canonical_disease_name": "idiopathic pulmonary fibrosis",
        "icd10_codes_str": "ICD10:J84.1",
        "characterized_diseases_index": {
            "EFO_0000768": {
                "efo_id": "EFO_0000768",
                "canonical_disease_name": "idiopathic pulmonary fibrosis",
                "icd10_codes_str": "ICD10:J84.1",
                "artifact_filename": "EFO_0000768_dossier.md",
                "targets_report": "Top target: TGFB1 (ENSG00000105329).",
                "treatments_report": "Approved drugs: Nintedanib (CHEMBL2103830), Pirfenidone.",
                "literature_report": "Recent peer-reviewed IPF study.",
                "dossier_markdown": "# IPF Dossier\nTargets: TGFB1 (ENSG00000105329). Drug: Nintedanib.",
            },
            "EFO_0000384": {
                "efo_id": "EFO_0000384",
                "canonical_disease_name": "Crohn's disease",
                "icd10_codes_str": "ICD10:K50",
                "artifact_filename": "EFO_0000384_dossier.md",
                "targets_report": "Top target: NOD2, TNF.",
                "treatments_report": "Approved drugs: Infliximab, Ustekinumab.",
                "literature_report": "Recent Crohn's mucosal immunology paper.",
                "dossier_markdown": "# Crohn's Disease Dossier\nTargets: NOD2, TNF. Drug: Infliximab.",
            },
        },
    }

    bg_task = consolidator.schedule_consolidation(
        state_snapshot=state_snapshot,
        app_name="app",
        user_id="researcher_1",
        session_id="sess_101",
        memory_service=vec_service,
    )
    assert not bg_task.done() or consolidator.completed_consolidations >= 0
    await consolidator.drain_background_tasks()
    assert consolidator.pending_task_count == 0
    assert consolidator.completed_consolidations == 2

    # 4. Re-open a fresh PersistentVectorMemoryService on the same SQLite file to verify cross-session persistence
    reopened_service = PersistentVectorMemoryService(db_path=db_file)
    search_resp = await reopened_service.search_memory(
        app_name="app",
        user_id="researcher_1",
        query="TGFB1 nintedanib pulmonary fibrosis",
    )
    assert len(search_resp.memories) >= 1
    top_memory = search_resp.memories[0]
    assert top_memory.custom_metadata["efo_id"] == "EFO_0000768"
    assert "TGFB1" in top_memory.content.parts[0].text

    # 5. Verify search_long_term_disease_memory tool contract and validation recovery_instructions
    with patch(
        "app.app_utils.services.get_memory_service",
        return_value=reopened_service,
    ):
        tool_res = await search_long_term_disease_memory(
            "idiopathic pulmonary fibrosis TGFB1", top_k=3
        )
        assert tool_res["status"] == "success"
        assert tool_res["returned_count"] >= 1
        assert tool_res["matches"][0]["efo_id"] == "EFO_0000768"

        bad_mem_query = await search_long_term_disease_memory("x", top_k=3)
        assert bad_mem_query["status"] == "validation_error"
        assert "RECOVERY:" in (bad_mem_query["recovery_instructions"] or "")


@pytest.mark.asyncio
async def test_strategic_model_routing_tiers_and_fallback() -> None:
    """Verify multi-tier role assignment (FAST_MODEL vs PRO_MODEL), dynamic complexity routing, and fallback."""
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from app.agent import (
        FAST_MODEL,
        PRO_MODEL,
        classify_query_complexity,
        disease_resolver_agent,
        dossier_synthesizer_agent,
        fallback_to_fast_model_on_error,
        literature_agent,
        route_model_by_complexity,
        targets_agent,
        treatments_agent,
    )

    # 1. Verify static role-based tiering across sub-agents
    assert FAST_MODEL != PRO_MODEL
    for extraction_agent in (
        disease_resolver_agent,
        targets_agent,
        treatments_agent,
        literature_agent,
    ):
        assert extraction_agent.model.model == FAST_MODEL  # type: ignore[union-attr]
        assert extraction_agent.generate_content_config.temperature == 0.0  # type: ignore[union-attr]

    assert dossier_synthesizer_agent.model.model == PRO_MODEL  # type: ignore[union-attr]
    assert dossier_synthesizer_agent.generate_content_config.temperature == 0.2  # type: ignore[union-attr]
    assert (
        dossier_synthesizer_agent.on_model_error_callback
        == fallback_to_fast_model_on_error
    )

    # 2. Verify dynamic complexity router (`route_model_by_complexity`)
    tier_fast, model_fast, _ = classify_query_complexity("Characterize IPF")
    assert tier_fast == "fast_extraction"
    assert model_fast == FAST_MODEL

    tier_pro, model_pro, reason_pro = classify_query_complexity(
        "Compare the shared targets and unmet need between IPF and Crohn's disease"
    )
    assert tier_pro == "deep_synthesis"
    assert model_pro == PRO_MODEL
    assert "comparative_or_analytical_intent" in reason_pro

    cb_ctx = MagicMock()
    cb_ctx.state = {}
    cb_ctx.agent_name = "disease_characterisation"
    cb_ctx.user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Compare target druggability across IPF vs ulcerative colitis"
            )
        ],
    )
    req = LlmRequest(model=FAST_MODEL, contents=[])
    await route_model_by_complexity(cb_ctx, req)
    assert req.model == PRO_MODEL
    assert req.config.temperature == 0.2
    assert cb_ctx.state["last_model_routing"]["tier"] == "deep_synthesis"

    # 3. Verify resilient fallback (`fallback_to_fast_model_on_error`) from PRO_MODEL -> FAST_MODEL
    expected_fallback_resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Synthesized fallback dossier")],
        )
    )

    async def _fake_generate(request: Any, stream: bool = False):
        assert request.model == FAST_MODEL
        yield expected_fallback_resp

    with patch.object(
        type(dossier_synthesizer_agent.model),
        "generate_content_async",
        side_effect=_fake_generate,
    ):
        recovered = await fallback_to_fast_model_on_error(
            cb_ctx,
            req,
            RuntimeError("404 Model not available in region"),
        )
        assert recovered is expected_fallback_resp
        assert req.model == FAST_MODEL



