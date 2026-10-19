# Future Improvements & Roadmap: Automated Disease Characterisation Agent

This document tracks planned extensions and enhancements for future iterations of the **Automated Disease Characterisation Agent**.

---

## 1. Seminal & High-Impact Literature Selection (Beyond Recency Only)
- **Current Behavior:** [`fetch_recent_publications`](disease-characterisation/app/tools.py) concurrently queries the Europe PMC REST API for peer-reviewed articles (`SRC:MED`) and `bioRxiv`/`medRxiv` preprints (`SRC:PPR`) sorted by publication date (`FIRST_PDATE_D desc`), interleaving them into a balanced cohort (up to 3 peer-reviewed + up to 2 preprints, strictly capped at $\le 5$).
- **Planned Improvement:**
  - Balance recency with **scientific impact and foundational importance** so the top $\le 5$ literature list includes **seminal landmark papers** alongside recent breakthroughs and preprints.
  - Incorporate **citation counts** (e.g., Europe PMC `citedByCount`, OpenAlex, or Semantic Scholar citation metrics) and **author affiliation signals** (e.g., leading academic medical centers, research institutes, and top universities).
  - Retrieve a hybrid cohort combining top highly cited seminal reviews/landmark studies with recent high-impact articles and `bioRxiv`/`medRxiv` preprints.

---

## 2. Patent Ownership & Exclusivity / Expiration Status for Approved Treatments
- **Current Behavior:** [`fetch_treatments_and_trials`](disease-characterisation/app/tools.py) retrieves approved and investigational drugs, ChEMBL IDs, modalities, mechanisms of action, targets, and maximum clinical trial phases.
- **Planned Improvement:**
  - For **approved treatments** (Max Phase 4), enrich the therapeutics table with **patent and exclusivity metadata**:
     - **Patent Owner / Market Authorization Holder** (originator / assignee company).
     - **Patent Expiration Date / Exclusivity Status** (or explicitly state if the therapy is **off-patent / generic / biosimilar available**).
  - **Candidate Public Data Sources:** FDA Orange Book / Purple Book APIs, ChEMBL / SureChEMBL patent mappings, or OpenFDA drug label/NDA sponsor records.

---

## 3. Unmet Medical Need & Strategic Opportunity Synthesis
- **Current Behavior:** [`dossier_synthesizer_agent`](disease-characterisation/app/agent.py) produces an Executive Scoping Summary focused on disease pathology, validated target classes, and active therapeutic mechanisms.
- **Planned Improvement:**
  - Add a dedicated **Unmet Need & Strategic Opportunities** section to the synthesized Disease Characterisation Dossier:
     - **Unmet Clinical Need Summary:** Limitations of current standard-of-care (e.g., incomplete efficacy, dose-limiting toxicities, refractory patient subpopulations, symptomatic-only relief, or diagnostic delays).
     - **Most Promising Opportunities:** Highlight actionable white-space opportunities for tackling the disease, clearly distinguishing between:
       - **Disease Modification:** Novel targets/pathways capable of halting or reversing underlying pathology (e.g., genetically validated targets with high tractability but no approved drugs).
       - **Disease / Symptom Management:** Opportunities to improve patient quality of life, tolerability, delivery modality, or combination regimens.
