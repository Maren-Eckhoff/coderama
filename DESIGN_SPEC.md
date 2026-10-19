# DESIGN_SPEC.md — Automated Disease Characterisation Agent

## Overview

The **Automated Disease Characterisation Agent** accelerates early-stage scoping in Life Sciences. Given a user-entered disease name, the agent normalizes the indication to canonical **EFO** (Experimental Factor Ontology) and **ICD-10** identifiers, executes parallel biomedical lookups for **known biological targets**, **existing treatments & clinical trials**, and **recent publications (including bioRxiv/medRxiv preprints)**, and synthesizes a grounded, citation-backed **Disease Characterisation Dossier**.

- **GCP Project ID**: `kaggle-sandbox-686020`
- **Default Region / Zone**: `us-east1` (Zone: `us-east1-d`)
- **Development Mode**: Local prototype first (`--prototype`) with **ADK Web Playground** (`agents-cli playground`)
- **Target Production Architecture**: **Vertex AI Agent Runtime** (`--deployment-target agent_runtime`) published to **Gemini Enterprise** (`agents-cli publish gemini-enterprise`)
- **Context Compaction**: Strict sliding-window compaction after every **5 events** (`EventsCompactionConfig(compaction_interval=5, overlap_size=1)`)
- **Strategic Multi-Tier Model Routing**:
  - **Tier 1 (`FAST_MODEL = "gemini-3.8-flash"`, `temperature=0.0`)**: Assigned to deterministic API extraction & normalization agents (`disease_resolver_agent`, `targets_agent`, `treatments_agent`, `literature_agent`) and fast single-disease coordinator turns.
  - **Tier 2 (`PRO_MODEL = "gemini-3.8-pro"`, `temperature=0.2`)**: Assigned to `dossier_synthesizer_agent` for cross-domain synthesis, and dynamically selected on `root_agent` via `before_model_callback` (`route_model_by_complexity`) for comparative, multi-disease, and mechanistic reasoning queries—protected by an automatic `on_model_error_callback` fallback (`fallback_to_fast_model_on_error`).
- **Long-Term Vector Memory & Retrieval**: Persistent vector store (`PersistentVectorMemoryService` backed by SQLite + 256-dim L2-normalized dense embeddings, with automatic delegation to `VertexAiRagMemoryService` (`VERTEX_AI_RAG_CORPUS`), `VertexAiMemoryBankService` (`GOOGLE_CLOUD_AGENT_ENGINE_ID`), and `VertexAiSearchTool` (`VERTEX_AI_SEARCH_DATA_STORE_ID`)), surfaced via `preload_memory`, `load_memory`, and `search_long_term_disease_memory`.
- **Async Background Memory Consolidation**: Non-blocking `asyncio.create_task` worker pool (`AsyncMemoryConsolidator` triggered via `after_agent_callback=trigger_async_memory_consolidation` and `save_dossier_artifact`) that asynchronously deduplicates, scrubs PII, embeds, and persists consolidated disease dossiers into long-term vector memory without adding user-facing latency.

---

## Architecture & Sub-Agents

```
root_agent (Coordinator LlmAgent)
└── disease_characterisation_pipeline (Workflow DAG)
    ├── 1. disease_resolver_agent (LlmAgent)
    │      └── Tool: resolve_disease_ontology -> writes session.state["disease_context"] (EFO ID + ICD-10 codes)
    ├── 2. Parallel Fan-Out: (targets_agent, treatments_agent, literature_agent)
    │      ├── targets_agent (LlmAgent)
    │      │   └── Tool: fetch_known_targets -> writes session.state["targets_report"]
    │      ├── treatments_agent (LlmAgent)
    │      │   └── Tool: fetch_treatments_and_trials -> writes session.state["treatments_report"]
    │      └── literature_agent (LlmAgent)
    │          └── Tool: fetch_recent_publications -> writes session.state["literature_report"]
    ├── 3. research_join (JoinNode)
    │      └── Synchronizes all three parallel research branches
    └── 4. dossier_synthesizer_agent (LlmAgent)
           └── Tool: save_dossier_artifact -> produces final structured Markdown dossier
```

---

## Example Use Cases

### Use Case 1: Full Disease Characterisation
- **Input**: `"Characterize Idiopathic Pulmonary Fibrosis"`
- **Expected Behavior**:
  1. `disease_resolver_agent` resolves `"Idiopathic Pulmonary Fibrosis"` to `EFO_0000768` and extracts its **ICD-10** code (`J84.1` / `ICD10:J84.1`).
  2. `parallel_research_agent` concurrently retrieves:
     - Top known targets (e.g., `TGFB1`, `FGFR`, `PDGFRB`, `MUC5B`, `TERT`) with association scores and tractability.
     - Existing treatments and clinical pipeline (e.g., *Nintedanib* [`CHEMBL2103830`], *Pirfenidone* [`CHEMBL12713`], plus active Phase II/III trials with `NCT` IDs).
     - Up to **5** recent publications from PubMed and `bioRxiv`/`medRxiv` with `PMID`/`DOI` links and source labels (`[Peer-Reviewed]` vs. `[Preprint - bioRxiv/medRxiv]`).
  3. `dossier_synthesizer_agent` outputs a structured Markdown dossier displaying the **EFO** and **ICD-10** codes in the header table, followed by Targets, Treatments, and Recent Publications (max 5).

### Use Case 2: Interactive Follow-Up in Session
- **Input**: `"Which of the known targets you found have small-molecule tractability?"`
- **Expected Behavior**:
  - `root_agent` answers directly from `session.state["targets_report"]` (or invokes `fetch_known_targets` for deeper target details) without re-running the entire pipeline, while `EventsCompactionConfig(compaction_interval=5)` compacts older events after 5 turns.

### Use Case 3: Sparse Literature or Rare Disease
- **Input**: `"Characterize Farber lipogranulomatosis"`
- **Expected Behavior**:
  - Resolves EFO and ICD-10 (`E75.2`), retrieves `ASAH1` target and known clinical trials, and returns fewer than 5 publications if fewer than 5 relevant recent papers/preprints exist—never fabricating citations to reach 5.

---

## Tools Required

1. **`resolve_disease_ontology(disease_query: str) -> dict`**
   - **Purpose**: Searches the Open Targets Platform GraphQL API (`search` + `disease(efoId)` `dbXRefs`) and EBI OLS4 REST API fallback to resolve a free-text disease name into its canonical `efo_id`, `name`, `description`, `synonyms`, `therapeutic_areas`, and **`icd10_codes`**.
   - **API**: `https://api.platform.opentargets.org/api/v4/graphql` & `https://www.ebi.ac.uk/ols4/api/search`
   - **Authentication**: None (public APIs).

2. **`fetch_known_targets(efo_id: str, limit: int) -> dict`**
   - **Purpose**: Queries Open Targets GraphQL (`disease.associatedTargets`) for top-ranked targets associated with `efo_id`, returning gene symbol, Ensembl ID (`ENSG*`), target name, overall association score, datatype scores (genetic, somatic, literature, pathways), and tractability summary.
   - **API**: `https://api.platform.opentargets.org/api/v4/graphql`
   - **Authentication**: None (public API).

3. **`fetch_treatments_and_trials(efo_id: str, disease_name: str, limit: int) -> dict`**
   - **Purpose**: Queries Open Targets GraphQL (`disease.knownDrugs`) for approved/investigational drugs (ChEMBL ID, drug name, drug type/modality, Mechanism of Action, target symbol, max clinical phase) and ClinicalTrials.gov REST API v2 (`/api/v2/studies`) for active/recruiting trials (`NCT` ID, title, phase, status, interventions).
   - **API**: `https://api.platform.opentargets.org/api/v4/graphql` & `https://clinicaltrials.gov/api/v2/studies`
   - **Authentication**: None (public APIs).

4. **`fetch_recent_publications(disease_name: str, max_results: int) -> dict`**
   - **Purpose**: Queries the Europe PMC REST API (`/europepmc/webservices/rest/search`) across both peer-reviewed literature (`SRC:MED`) and preprints (`SRC:PPR`, filtering/tagging `bioRxiv` and `medRxiv`). Enforces a hard cap of **`max_results <= 5`** (returning fewer if fewer exist). Returns title, authors, journal/preprint server (`bioRxiv` / `medRxiv`), publication year, `pmid`, `doi`, URL, and abstract snippet.
   - **API**: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
   - **Authentication**: None (public API).

5. **`save_dossier_artifact(efo_id: str, markdown_content: str, tool_context: ToolContext) -> SaveDossierArtifactOutput`**
   - **Purpose**: Persists the synthesized Markdown dossier into the ADK `ArtifactService` (`{efo_id}_dossier.md`), updates the in-session multi-disease consolidation index, and schedules a non-blocking `asyncio.create_task` background job (`AsyncMemoryConsolidator`) to embed and store the dossier in long-term vector memory.

6. **`search_long_term_disease_memory(query: str, top_k: int, tool_context: ToolContext) -> SearchLongTermMemoryOutput`** (plus ADK `preload_memory`, `load_memory`, and optional `VertexAiSearchTool`)
   - **Purpose**: Executes semantic cosine-similarity + lexical hybrid search over the persistent vector store (`PersistentVectorMemoryService` / `VertexAiRagMemoryService` / `VertexAiMemoryBankService`) to retrieve previously consolidated disease dossiers, targets, and trials across sessions.

---

## Constraints & Safety Rules

1. **Zero Hallucinated Identifiers**: Every EFO code, ICD-10 code, gene symbol, ChEMBL drug ID, ClinicalTrials.gov `NCT` ID, `PMID`, and `DOI` in the final response MUST originate from a tool response in the current session.
2. **Publication Cap & Preprint Transparency**: Surface **at most 5 publications** (fewer if fewer relevant papers exist). Every preprint from `bioRxiv` or `medRxiv` must be explicitly labeled `[Preprint - bioRxiv]` or `[Preprint - medRxiv]` so researchers distinguish non-peer-reviewed preprints from peer-reviewed journal articles.
3. **EFO & ICD-10 Visibility**: Always display the normalized **EFO ID** and **ICD-10 code(s)** in the top summary card of the frontend output (or explicitly state `"ICD-10: Not mapped in ontology cross-references"` if an ultra-rare indication lacks an ICD-10 mapping).
4. **Strict Context Compaction**: Configure `EventsCompactionConfig(compaction_interval=5, overlap_size=1)` on the ADK `App` to compact history after every 5 events.
5. **Non-Clinical Research Disclaimer**: Every characterisation dossier must include a standard scoping disclaimer stating that the output is for *Life Sciences research and target scoping only and does not constitute medical or clinical advice*.

---

## Success Criteria

1. **Ontology Accuracy**: Given standard disease names or acronyms (e.g., *IPF*, *NSCLC*, *Crohn's disease*, *Alzheimer's disease*), the agent resolves both a valid `EFO_*`/`MONDO_*` ID and corresponding `ICD-10` code(s).
2. **Complete Three-Pillar Coverage**: Dossier includes structured sections for (a) Known Targets with scores, (b) Existing Treatments & Clinical Trials with MoA/phases, and (c) Recent Publications (≤5, supporting PubMed + bioRxiv/medRxiv).
3. **Unit & Eval Thresholds**:
   - All `pytest` unit tests for biomedical API tools pass (`100%`).
   - ADK evaluation suite (`agents-cli eval run`) meets configured thresholds for tool trajectory and grounded response quality.
4. **Observability**: OpenTelemetry / Cloud Trace spans (`invocation`, `agent_run`, `call_llm`, `execute_tool`) are emitted and verifiable.

---

## Reference Samples

- **`deep-search`** (`adk-samples/python/agents/deep-search`): Reference pattern for multi-agent research orchestration (`SequentialAgent` / `ParallelAgent`), state handoff via `output_key`, and grounded source citations.
