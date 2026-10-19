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

from dotenv import load_dotenv
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models import Gemini
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.genai import types

load_dotenv()

from app.app_utils.telemetry import DiseaseObservabilityPlugin
from app.tools import (
    fetch_known_targets,
    fetch_recent_publications,
    fetch_treatments_and_trials,
    resolve_disease_ontology,
    save_dossier_artifact,
)

MODEL = "gemini-3.8-flash"

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
        "disease_context": "Not yet resolved.",
        "targets_report": "Not yet retrieved.",
        "treatments_report": "Not yet retrieved.",
        "literature_report": "Not yet retrieved.",
    }
    for key, default_val in defaults.items():
        if key not in state:
            state[key] = default_val


def _build_model() -> Gemini:
    """Constructs the Gemini model instance with standard retry options."""
    return Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


disease_resolver_agent = Agent(
    name="disease_resolver_agent",
    model=_build_model(),
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
Never invent EFO or ICD-10 codes that were not returned by the tool.""",
    tools=[resolve_disease_ontology],
    output_key="disease_context",
)

targets_agent = Agent(
    name="targets_agent",
    model=_build_model(),
    description="Fetches top known biological targets and tractability evidence for the resolved disease.",
    instruction="""You are the Target Discovery Specialist.
Using the resolved disease context below:
{disease_context}

1. Extract the canonical `efo_id` and call `fetch_known_targets(efo_id=..., limit=10)`.
2. Summarize the top associated biological targets including:
   - Gene Symbol and Ensembl ID (`ENSG...`)
   - Target Name
   - Overall Association Score and key datatype scores (genetic, somatic, known drug, literature, pathways)
   - Positive Tractability / Druggability modalities (Small Molecule, Antibody, PROTAC, etc.)
Never fabricate target symbols, Ensembl IDs, or scores.""",
    tools=[fetch_known_targets],
    output_key="targets_report",
)

treatments_agent = Agent(
    name="treatments_agent",
    model=_build_model(),
    description="Fetches existing approved/investigational treatments and active clinical trials for the disease.",
    instruction="""You are the Clinical & Therapeutics Landscape Specialist.
Using the resolved disease context below:
{disease_context}

1. Extract the canonical `efo_id` and canonical disease `name`, then call `fetch_treatments_and_trials(efo_id=..., disease_name=..., limit=10)`.
2. Summarize:
   - Approved and investigational drugs (`drug_name`, `chembl_id`, modality `drug_type`, `max_clinical_phase`, `mechanism_of_action`, and `target_symbol`)
   - Active or recent clinical trials from ClinicalTrials.gov (`nct_id`, `title`, `phases`, `overall_status`, `interventions`, and `lead_sponsor`)
Never fabricate ChEMBL IDs, drug names, or NCT trial identifiers.""",
    tools=[fetch_treatments_and_trials],
    output_key="treatments_report",
)

literature_agent = Agent(
    name="literature_agent",
    model=_build_model(),
    description="Surfaces up to 5 recent peer-reviewed publications and bioRxiv/medRxiv preprints.",
    instruction="""You are the Biomedical Literature & Preprint Specialist.
Using the resolved disease context below:
{disease_context}

1. Extract the canonical disease `name` and call `fetch_recent_publications(disease_name=..., max_results=5)`.
2. Report AT MOST 5 publications (if the tool returns fewer than 5, report only those returned—never invent papers to reach 5).
3. For each publication, preserve:
   - Source badge: `[Peer-Reviewed]` or `[Preprint - bioRxiv]` / `[Preprint - medRxiv]`
   - Title, Authors, Venue/Server, and Publication Year
   - Identifier (`PMID` and/or `DOI`) and URL
   - Concise abstract takeaway.""",
    tools=[fetch_recent_publications],
    output_key="literature_report",
)

parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    description="Concurrently retrieves known targets, existing treatments/trials, and recent literature/preprints.",
    sub_agents=[targets_agent, treatments_agent, literature_agent],
)

dossier_synthesizer_agent = Agent(
    name="dossier_synthesizer_agent",
    model=_build_model(),
    description="Synthesizes the grounded Disease Characterisation Dossier and saves it as a downloadable artifact.",
    instruction=f"""You are the Lead Life Sciences Scoping Synthesizer.
Combine the findings from the specialist agents into a clear, executive **Disease Characterisation Dossier**:

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
   - Render a prominent summary table at the very top showing **Disease Name**, **EFO ID**, **ICD-10 Code(s)**, **Therapeutic Areas**, and **Open Targets Link**.
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

Finally, call `save_dossier_artifact` with the `efo_id` and the complete Markdown dossier text, and present the full Markdown dossier to the user.""",
    tools=[save_dossier_artifact],
    output_key="characterisation_dossier",
)

disease_characterisation_pipeline = SequentialAgent(
    name="disease_characterisation_pipeline",
    description="Executes full disease characterisation: ontology normalization (EFO + ICD-10), parallel research across targets, treatments, and literature/preprints, and dossier synthesis.",
    sub_agents=[
        disease_resolver_agent,
        parallel_research_agent,
        dossier_synthesizer_agent,
    ],
)

root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="disease_characterisation",
    model=_build_model(),
    description="Automated Disease Characterisation Agent for Life Sciences scoping.",
    instruction="""You are the Automated Disease Characterisation Coordinator for Life Sciences scoping.

- When the user enters a disease name or asks to characterize/scope a disease, delegate to `disease_characterisation_pipeline` to run the full ontology normalization (EFO + ICD-10), parallel target/treatment/literature analysis, and dossier synthesis.
- When the user asks a targeted follow-up question about a disease already characterized in the current session (stored in `{disease_context}`, `{targets_report}`, `{treatments_report}`, `{literature_report}`), answer using the session state or call the specific tool (`resolve_disease_ontology`, `fetch_known_targets`, `fetch_treatments_and_trials`, `fetch_recent_publications`, `save_dossier_artifact`) as needed.
- Always ground every identifier (`EFO`, `ICD-10`, `ENSG`, `ChEMBL`, `NCT`, `PMID`, `DOI`) in tool outputs, cap publication lists at a maximum of 5 (labeling `bioRxiv`/`medRxiv` preprints clearly), and include a non-clinical research scoping disclaimer.""",
    sub_agents=[disease_characterisation_pipeline],
    tools=[
        resolve_disease_ontology,
        fetch_known_targets,
        fetch_treatments_and_trials,
        fetch_recent_publications,
        save_dossier_artifact,
    ],
    before_agent_callback=initialize_session_state,
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
