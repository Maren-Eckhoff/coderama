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

"""Strict Pydantic JSON Schemas for biomedical tool inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T", bound="StrictSchemaModel")

ONTOLOGY_ID_PATTERN = r"^(EFO|MONDO|ORPHA|Orphanet|HP|OTAR|DOID)[:_][0-9]+$"


class StrictSchemaModel(BaseModel):
    """Base Pydantic model enforcing strict JSON Schema validation (additionalProperties: false)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def validate_and_dump(model_cls: type[T], payload: dict[str, Any]) -> dict[str, Any]:
    """Strictly validates a dictionary against `model_cls` and returns a clean dict."""
    validated = model_cls.model_validate(payload)
    return validated.model_dump()


# ============================================================================
# 1. Input Schemas (Strict Input Validation)
# ============================================================================


class ResolveDiseaseInput(StrictSchemaModel):
    """Strict input schema for `resolve_disease_ontology`."""

    disease_query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "Disease name, medical acronym (e.g., 'IPF', 'NSCLC'), or canonical "
            "ontology ID (e.g., 'EFO_0000768', 'MONDO_0005148')."
        ),
    )


class FetchKnownTargetsInput(StrictSchemaModel):
    """Strict input schema for `fetch_known_targets`."""

    efo_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=ONTOLOGY_ID_PATTERN,
        description="Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of top associated targets to return (1 to 25).",
    )

    @field_validator("efo_id", mode="before")
    @classmethod
    def _normalize_efo_separator(cls, val: Any) -> str:
        if isinstance(val, str):
            return val.strip().replace(":", "_")
        return str(val or "")

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, val: Any) -> int:
        parsed = int(val)
        if parsed < 1:
            raise ValueError("limit must be >= 1")
        return min(parsed, 25)


class FetchTreatmentsAndTrialsInput(StrictSchemaModel):
    """Strict input schema for `fetch_treatments_and_trials`."""

    efo_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=ONTOLOGY_ID_PATTERN,
        description="Canonical ontology identifier (e.g., 'EFO_0000768' or 'MONDO_0005148').",
    )
    disease_name: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Canonical disease name used for querying ClinicalTrials.gov.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of drugs and clinical trials to return (1 to 25).",
    )

    @field_validator("efo_id", mode="before")
    @classmethod
    def _normalize_efo_separator(cls, val: Any) -> str:
        if isinstance(val, str):
            return val.strip().replace(":", "_")
        return str(val or "")

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, val: Any) -> int:
        parsed = int(val)
        if parsed < 1:
            raise ValueError("limit must be >= 1")
        return min(parsed, 25)


class FetchRecentPublicationsInput(StrictSchemaModel):
    """Strict input schema for `fetch_recent_publications`."""

    disease_name: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Canonical disease name to search in Europe PMC (PubMed + bioRxiv/medRxiv).",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum number of publications to return (strictly capped at <= 5).",
    )

    @field_validator("max_results", mode="before")
    @classmethod
    def _clamp_max_results(cls, val: Any) -> int:
        parsed = int(val)
        if parsed < 1:
            raise ValueError("max_results must be >= 1")
        return min(parsed, 5)


class SaveDossierArtifactInput(StrictSchemaModel):
    """Strict input schema for `save_dossier_artifact`."""

    efo_id: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Canonical EFO/MONDO identifier for the characterized disease.",
    )
    markdown_content: str = Field(
        ...,
        min_length=15,
        description="Complete Markdown text of the synthesized Disease Characterisation Dossier.",
    )


# ============================================================================
# 2. Output Schemas (Strict Output Validation & Recovery Contracts)
# ============================================================================


class TherapeuticAreaItem(StrictSchemaModel):
    """A therapeutic area ontology entry associated with a disease."""

    id: str | None = None
    name: str


class CandidateDiseaseItem(StrictSchemaModel):
    """A candidate disease ontology match used for alternative matches or disambiguation."""

    efo_id: str
    name: str
    description: str = ""


class DiseaseOntologyOutput(StrictSchemaModel):
    """Strict output schema for `resolve_disease_ontology`."""

    status: Literal["success", "ambiguous", "not_found", "validation_error", "error"]
    resolution_confidence: str = "not_found"
    requires_human_disambiguation: bool = False
    query: str = ""
    efo_id: str | None = None
    name: str | None = None
    icd10_codes: list[str] = Field(default_factory=list)
    other_xrefs: list[str] = Field(default_factory=list)
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    therapeutic_areas: list[TherapeuticAreaItem] = Field(default_factory=list)
    candidates: list[CandidateDiseaseItem] = Field(default_factory=list)
    alternative_matches: list[CandidateDiseaseItem] = Field(default_factory=list)
    open_targets_url: str | None = None
    message: str | None = None
    recovery_instructions: str | None = Field(
        default=None,
        description="Explicit step-by-step instructions for recovering from non-success or ambiguous states.",
    )


class TargetEvidenceItem(StrictSchemaModel):
    """A ranked biological target associated with a disease in Open Targets."""

    gene_symbol: str | None = None
    ensembl_id: str
    target_name: str | None = None
    biotype: str | None = None
    overall_association_score: float = Field(..., ge=0.0, le=1.0)
    datatype_scores: dict[str, float] = Field(default_factory=dict)
    tractability_positive_modalities: list[str] = Field(default_factory=list)
    open_targets_url: str


class KnownTargetsOutput(StrictSchemaModel):
    """Strict output schema for `fetch_known_targets`."""

    status: Literal["success", "not_found", "validation_error", "error"]
    efo_id: str = ""
    disease_name: str | None = None
    total_associated_targets: int = Field(default=0, ge=0)
    targets: list[TargetEvidenceItem] = Field(default_factory=list)
    message: str | None = None
    recovery_instructions: str | None = Field(
        default=None,
        description="Explicit step-by-step instructions for recovering from errors or zero-target results.",
    )


class KnownDrugItem(StrictSchemaModel):
    """An approved or clinical-stage therapeutic compound from Open Targets or ChEMBL."""

    drug_name: str | None = None
    chembl_id: str
    drug_type: str | None = None
    max_clinical_phase: float | int | None = None
    trial_status: str | None = None
    mechanism_of_action: str | None = None
    target_symbol: str | None = None
    target_ensembl_id: str | None = None
    chembl_url: str | None = None


class ClinicalTrialItem(StrictSchemaModel):
    """A clinical study record from ClinicalTrials.gov REST API v2."""

    nct_id: str
    title: str | None = None
    overall_status: str | None = None
    phases: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    lead_sponsor: str | None = None
    url: str | None = None


class TreatmentsAndTrialsOutput(StrictSchemaModel):
    """Strict output schema for `fetch_treatments_and_trials`."""

    status: Literal["success", "validation_error", "error"]
    efo_id: str = ""
    disease_name: str = ""
    unique_known_drugs_count: int = Field(default=0, ge=0)
    known_drugs: list[KnownDrugItem] = Field(default_factory=list)
    clinical_trials: list[ClinicalTrialItem] = Field(default_factory=list)
    partial_warnings: list[str] = Field(default_factory=list)
    message: str | None = None
    recovery_instructions: str | None = Field(
        default=None,
        description="Explicit step-by-step instructions for handling partial source warnings or empty pipelines.",
    )


class PublicationItem(StrictSchemaModel):
    """A peer-reviewed journal article (PubMed) or preprint (bioRxiv/medRxiv) from Europe PMC."""

    title: str | None = None
    authors: str | None = None
    venue: str
    publication_year: str | None = None
    first_publication_date: str | None = None
    publication_type: str
    source_badge: Literal[
        "[Peer-Reviewed]",
        "[Preprint - bioRxiv]",
        "[Preprint - medRxiv]",
        "[Preprint - bioRxiv/medRxiv]",
        "[Preprint - Preprint Server]",
    ] | str
    is_preprint: bool
    pmid: str | None = None
    doi: str | None = None
    cited_by_count: int = Field(default=0, ge=0)
    url: str
    abstract_snippet: str = ""


class RecentPublicationsOutput(StrictSchemaModel):
    """Strict output schema for `fetch_recent_publications`."""

    status: Literal["success", "validation_error", "error"]
    disease_name: str = ""
    max_allowed: int = Field(default=5, ge=1, le=5)
    returned_count: int = Field(default=0, ge=0, le=5)
    includes_preprints_from: list[str] = Field(
        default_factory=lambda: ["bioRxiv", "medRxiv"]
    )
    publications: list[PublicationItem] = Field(
        default_factory=list, max_length=5
    )
    message: str | None = None
    recovery_instructions: str | None = Field(
        default=None,
        description="Explicit step-by-step instructions when fewer than 5 papers exist or an upstream error occurs.",
    )


class SaveDossierArtifactOutput(StrictSchemaModel):
    """Strict output schema for `save_dossier_artifact`."""

    status: Literal["success", "saved_in_session_state", "validation_error"]
    artifact_filename: str = ""
    version: int = Field(default=0, ge=0)
    message: str | None = None
    recovery_instructions: str | None = Field(
        default=None,
        description="Explicit instructions when artifact persistence falls back to session state.",
    )
