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

"""Persistent Vector Memory Service, Vertex AI Search integration, and Async Background Memory Consolidation."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.adk.tools import ToolContext
from google.genai import types
from opentelemetry import trace
from pydantic import Field, ValidationError

from app.app_utils.telemetry import get_structured_logger, redact_pii
from app.schemas import (
    SearchLongTermMemoryInput,
    SearchLongTermMemoryOutput,
    validate_and_dump,
)

logger = get_structured_logger("disease_characterisation.memory")
tracer = trace.get_tracer("disease_characterisation.memory")

EMBEDDING_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9_:-]{2,}", re.IGNORECASE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "not",
    "yet",
    "have",
    "has",
}


def compute_text_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Computes an L2-normalized dense embedding vector for semantic similarity retrieval.

    Combines biomedical identifier tokens, word unigrams/bigrams, and character 3-grams
    using signed feature hashing (Weinberger et al.) into a dense `dim`-dimensional vector.
    """
    vec = [0.0] * dim
    cleaned = (text or "").lower().strip()
    if not cleaned:
        return vec

    tokens = [
        tok for tok in _TOKEN_RE.findall(cleaned) if tok not in _STOPWORDS
    ]
    if not tokens:
        tokens = [cleaned]

    features: list[tuple[str, float]] = []
    for idx, tok in enumerate(tokens):
        # Boost canonical biomedical identifiers (EFO, MONDO, ICD10, ENSG, CHEMBL, NCT)
        weight = (
            2.5
            if tok.startswith(
                ("efo_", "mondo_", "icd10:", "ensg", "chembl", "nct")
            )
            else 1.0
        )
        features.append((f"u:{tok}", weight))
        if idx + 1 < len(tokens):
            features.append((f"b:{tok}_{tokens[idx + 1]}", 1.2))
        if len(tok) >= 4:
            for c_idx in range(len(tok) - 2):
                features.append((f"g:{tok[c_idx : c_idx + 3]}", 0.35))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h_val = int.from_bytes(digest, "little")
        bucket = h_val % dim
        sign = 1.0 if ((h_val >> 32) & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Computes cosine similarity between two L2-normalized vectors in [-1.0, 1.0]."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(-1.0, min(1.0, float(dot)))


class PersistentVectorMemoryService(BaseMemoryService):
    """Persistent hybrid vector + lexical long-term memory service implementing ADK `BaseMemoryService`.

    Persists consolidated disease dossiers and session events in a local SQLite vector store
    while optionally delegating to `VertexAiRagMemoryService` (when `VERTEX_AI_RAG_CORPUS` is set)
    or `VertexAiMemoryBankService` (when `GOOGLE_CLOUD_AGENT_ENGINE_ID` is set).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        rag_corpus: str | None = None,
        agent_engine_id: str | None = None,
        project: str | None = None,
        location: str | None = None,
        similarity_top_k: int = 5,
        vector_similarity_threshold: float = 0.08,
    ) -> None:
        if db_path is None:
            env_db = os.environ.get("VECTOR_MEMORY_DB_PATH")
            if env_db:
                db_path = Path(env_db)
            else:
                base_dir = Path(__file__).resolve().parent.parent / ".adk"
                base_dir.mkdir(parents=True, exist_ok=True)
                db_path = base_dir / "vector_memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_top_k = similarity_top_k
        self.vector_similarity_threshold = vector_similarity_threshold
        self._init_db()

        self._cloud_delegate: BaseMemoryService | None = None
        resolved_rag = (
            rag_corpus
            or os.environ.get("VERTEX_AI_RAG_CORPUS")
            or os.environ.get("RAG_CORPUS")
        )
        resolved_engine_id = agent_engine_id or os.environ.get(
            "GOOGLE_CLOUD_AGENT_ENGINE_ID"
        )
        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        resolved_location = (
            location
            or os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
        )

        if resolved_rag:
            try:
                from google.adk.memory import VertexAiRagMemoryService

                self._cloud_delegate = VertexAiRagMemoryService(
                    rag_corpus=resolved_rag,
                    similarity_top_k=similarity_top_k,
                    project=resolved_project,
                    location=resolved_location,
                )
            except Exception as exc:
                logger.warning(
                    "Could not initialize VertexAiRagMemoryService (%s); using local persistent vector store.",
                    exc,
                )
        elif resolved_engine_id:
            try:
                from google.adk.memory import VertexAiMemoryBankService

                self._cloud_delegate = VertexAiMemoryBankService(
                    project=resolved_project,
                    location=resolved_location,
                    agent_engine_id=resolved_engine_id.split("/")[-1],
                )
            except Exception as exc:
                logger.warning(
                    "Could not initialize VertexAiMemoryBankService (%s); using local persistent vector store.",
                    exc,
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consolidated_vector_memories (
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    efo_id TEXT,
                    disease_name TEXT,
                    author TEXT,
                    timestamp TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    PRIMARY KEY (app_name, user_id, memory_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vector_memories_scope
                ON consolidated_vector_memories (app_name, user_id, efo_id)
                """
            )
            conn.commit()

    def _upsert_record_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        memory_key: str,
        efo_id: str | None,
        disease_name: str | None,
        author: str | None,
        timestamp: str,
        content_text: str,
        metadata: Mapping[str, Any],
    ) -> None:
        scrubbed_text = redact_pii(content_text)
        embedding = compute_text_embedding(
            f"{efo_id or ''} {disease_name or ''} {scrubbed_text}"
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO consolidated_vector_memories (
                    app_name, user_id, memory_key, efo_id, disease_name,
                    author, timestamp, content_text, metadata_json, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(app_name, user_id, memory_key) DO UPDATE SET
                    efo_id = excluded.efo_id,
                    disease_name = excluded.disease_name,
                    author = excluded.author,
                    timestamp = excluded.timestamp,
                    content_text = excluded.content_text,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json
                """,
                (
                    app_name or "app",
                    user_id or "default_user",
                    memory_key,
                    efo_id,
                    disease_name,
                    author or "disease_characterisation",
                    timestamp,
                    scrubbed_text,
                    json.dumps(dict(metadata), default=str),
                    json.dumps(embedding),
                ),
            )
            conn.commit()

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Persists explicit `MemoryEntry` objects into the vector store and optional cloud backend."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for idx, entry in enumerate(memories):
            parts = (entry.content.parts if entry.content else None) or []
            text = "\n".join(p.text for p in parts if getattr(p, "text", None))
            if not text.strip():
                continue
            merged_meta: dict[str, Any] = {}
            if custom_metadata:
                merged_meta.update(custom_metadata)
            if entry.custom_metadata:
                merged_meta.update(entry.custom_metadata)

            efo_id = str(merged_meta.get("efo_id") or "") or None
            disease_name = str(merged_meta.get("disease_name") or "") or None
            memory_key = (
                entry.id
                or (f"dossier:{efo_id}" if efo_id else None)
                or hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            )
            await asyncio.to_thread(
                self._upsert_record_sync,
                app_name=app_name,
                user_id=user_id,
                memory_key=memory_key,
                efo_id=efo_id,
                disease_name=disease_name,
                author=entry.author or "disease_characterisation",
                timestamp=entry.timestamp or now_iso,
                content_text=text,
                metadata=merged_meta,
            )

        if self._cloud_delegate is not None:
            try:
                await self._cloud_delegate.add_memory(
                    app_name=app_name,
                    user_id=user_id,
                    memories=memories,
                    custom_metadata=custom_metadata,
                )
            except Exception as exc:
                logger.debug("Cloud memory delegate add_memory skipped: %s", exc)

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Ingests event deltas into the persistent vector store and optional cloud backend."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entries: list[MemoryEntry] = []
        for ev in events:
            if not ev.content or not ev.content.parts:
                continue
            texts = [
                p.text
                for p in ev.content.parts
                if getattr(p, "text", None) and len(p.text.strip()) >= 20
            ]
            if not texts:
                continue
            joined = redact_pii("\n".join(texts))
            ev_id = getattr(ev, "id", None) or hashlib.sha256(
                joined.encode("utf-8")
            ).hexdigest()[:20]
            entries.append(
                MemoryEntry(
                    id=f"event:{session_id or 'session'}:{ev_id}",
                    author=getattr(ev, "author", None) or "agent",
                    timestamp=now_iso,
                    content=types.Content(
                        role="model", parts=[types.Part.from_text(text=joined)]
                    ),
                    custom_metadata=dict(custom_metadata or {}),
                )
            )
        if entries:
            await self.add_memory(
                app_name=app_name,
                user_id=user_id,
                memories=entries,
                custom_metadata=custom_metadata,
            )
        if self._cloud_delegate is not None:
            try:
                await self._cloud_delegate.add_events_to_memory(
                    app_name=app_name,
                    user_id=user_id,
                    events=events,
                    session_id=session_id,
                    custom_metadata=custom_metadata,
                )
            except Exception as exc:
                logger.debug(
                    "Cloud memory delegate add_events_to_memory skipped: %s",
                    exc,
                )

    async def add_session_to_memory(self, session: Session) -> None:
        """Extracts consolidated disease knowledge and events from `session` into persistent vector memory."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state = getattr(session, "state", {}) or {}
        app_name = getattr(session, "app_name", None) or "app"
        user_id = getattr(session, "user_id", None) or "default_user"
        session_id = getattr(session, "id", None) or "session"

        entries = build_consolidated_memory_entries_from_state(
            state, timestamp=now_iso
        )
        if entries:
            await self.add_memory(
                app_name=app_name,
                user_id=user_id,
                memories=entries,
                custom_metadata={
                    "session_id": session_id,
                    "wait_for_completion": False,
                    "disable_consolidation": False,
                },
            )
        elif getattr(session, "events", None):
            await self.add_events_to_memory(
                app_name=app_name,
                user_id=user_id,
                events=session.events,
                session_id=session_id,
            )

        if self._cloud_delegate is not None:
            try:
                await self._cloud_delegate.add_session_to_memory(session)
            except Exception as exc:
                logger.debug(
                    "Cloud memory delegate add_session_to_memory skipped: %s",
                    exc,
                )

    def _search_records_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        q_clean = (query or "").strip()
        if not q_clean:
            return []
        q_vec = compute_text_embedding(q_clean)
        q_tokens = {
            tok
            for tok in _TOKEN_RE.findall(q_clean.lower())
            if tok not in _STOPWORDS
        }

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_key, efo_id, disease_name, author, timestamp,
                       content_text, metadata_json, embedding_json
                FROM consolidated_vector_memories
                WHERE app_name = ? AND (user_id = ? OR ? = 'default_user' OR user_id = 'default_user')
                """,
                (
                    app_name or "app",
                    user_id or "default_user",
                    user_id or "default_user",
                ),
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            try:
                doc_vec = json.loads(row["embedding_json"])
            except Exception:
                doc_vec = []
            cos_sim = cosine_similarity(q_vec, doc_vec)
            doc_text_lower = (
                f"{row['efo_id'] or ''} {row['disease_name'] or ''} {row['content_text']}"
            ).lower()
            lexical_hits = sum(1 for tok in q_tokens if tok in doc_text_lower)
            lexical_boost = (
                0.25 * (lexical_hits / len(q_tokens)) if q_tokens else 0.0
            )
            hybrid_score = min(1.0, max(-1.0, cos_sim + lexical_boost))

            if (
                hybrid_score >= self.vector_similarity_threshold
                or lexical_hits > 0
            ):
                scored.append(
                    (
                        round(hybrid_score, 4),
                        {
                            "memory_key": row["memory_key"],
                            "efo_id": row["efo_id"],
                            "disease_name": row["disease_name"],
                            "author": row["author"],
                            "timestamp": row["timestamp"],
                            "content_text": row["content_text"],
                            "metadata": json.loads(
                                row["metadata_json"] or "{}"
                            ),
                        },
                    )
                )

        scored.sort(
            key=lambda item: (item[0], item[1].get("timestamp") or ""),
            reverse=True,
        )
        return scored[:top_k]

    async def search_with_scores(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[float, MemoryEntry]]:
        """Searches persistent vector memory and returns `(similarity_score, MemoryEntry)` pairs."""
        limit = top_k or self.similarity_top_k
        with tracer.start_as_current_span("memory.vector_search") as span:
            span.set_attribute("memory.query", redact_pii(query))
            span.set_attribute("memory.top_k", limit)
            raw_matches = await asyncio.to_thread(
                self._search_records_sync,
                app_name=app_name,
                user_id=user_id,
                query=query,
                top_k=limit,
            )
            results: list[tuple[float, MemoryEntry]] = []
            for score, rec in raw_matches:
                meta = dict(rec.get("metadata") or {})
                if rec.get("efo_id"):
                    meta["efo_id"] = rec["efo_id"]
                if rec.get("disease_name"):
                    meta["disease_name"] = rec["disease_name"]
                meta["similarity_score"] = score
                results.append(
                    (
                        score,
                        MemoryEntry(
                            id=rec["memory_key"],
                            author=rec.get("author")
                            or "disease_characterisation",
                            timestamp=rec.get("timestamp"),
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=rec["content_text"]
                                    )
                                ],
                            ),
                            custom_metadata=meta,
                        ),
                    )
                )
            span.set_attribute("memory.returned_count", len(results))
            return results

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """ADK `BaseMemoryService.search_memory` entrypoint combining local vector store and cloud delegate."""
        scored_entries = await self.search_with_scores(
            app_name=app_name,
            user_id=user_id,
            query=query,
            top_k=self.similarity_top_k,
        )
        memories = [entry for _, entry in scored_entries]

        if self._cloud_delegate is not None:
            try:
                cloud_resp = await self._cloud_delegate.search_memory(
                    app_name=app_name,
                    user_id=user_id,
                    query=query,
                )
                seen_ids = {m.id for m in memories if m.id}
                for cm in cloud_resp.memories:
                    if not cm.id or cm.id not in seen_ids:
                        memories.append(cm)
            except Exception as exc:
                logger.debug(
                    "Cloud memory delegate search_memory skipped: %s", exc
                )

        return SearchMemoryResponse(memories=memories[: self.similarity_top_k])


def build_consolidated_memory_entries_from_state(
    state: Mapping[str, Any],
    *,
    timestamp: str | None = None,
) -> list[MemoryEntry]:
    """Extracts and deduplicates structured long-term `MemoryEntry` objects from session state."""
    now_iso = (
        timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    entries: list[MemoryEntry] = []
    seen_efos: set[str] = set()

    index = state.get("characterized_diseases_index")
    if isinstance(index, Mapping):
        for efo_key, record in index.items():
            if not isinstance(record, Mapping):
                continue
            efo_id = str(record.get("efo_id") or efo_key).strip()
            if not efo_id or efo_id == "Not yet resolved.":
                continue
            seen_efos.add(efo_id)
            disease_name = str(
                record.get("canonical_disease_name") or efo_id
            ).strip()
            icd10_str = str(
                record.get("icd10_codes_str")
                or "ICD-10: Not mapped in ontology cross-references"
            )
            dossier_md = redact_pii(str(record.get("dossier_markdown") or ""))
            targets_rep = redact_pii(str(record.get("targets_report") or ""))
            treatments_rep = redact_pii(
                str(record.get("treatments_report") or "")
            )
            literature_rep = redact_pii(
                str(record.get("literature_report") or "")
            )

            consolidated_text = (
                f"[Consolidated Long-Term Disease Memory] {disease_name} "
                f"(EFO ID: {efo_id} | ICD-10: {icd10_str})\n"
                f"Dossier Summary:\n{dossier_md or targets_rep}"
            ).strip()
            entries.append(
                MemoryEntry(
                    id=f"dossier:{efo_id}",
                    author="disease_characterisation",
                    timestamp=now_iso,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=consolidated_text)],
                    ),
                    custom_metadata={
                        "memory_type": "consolidated_disease_dossier",
                        "efo_id": efo_id,
                        "disease_name": disease_name,
                        "icd10_codes": icd10_str,
                        "has_targets": bool(targets_rep),
                        "has_treatments": bool(treatments_rep),
                        "has_literature": bool(literature_rep),
                    },
                )
            )

    current_efo = str(state.get("efo_id") or "").strip()
    if (
        current_efo
        and current_efo != "Not yet resolved."
        and current_efo not in seen_efos
        and state.get("resolution_status") == "success"
    ):
        disease_name = str(
            state.get("canonical_disease_name") or current_efo
        ).strip()
        icd10_str = str(
            state.get("icd10_codes_str")
            or "ICD-10: Not mapped in ontology cross-references"
        )
        dossier_md = redact_pii(
            str(
                state.get("latest_dossier_markdown")
                or state.get("characterisation_dossier")
                or ""
            )
        )
        targets_rep = redact_pii(str(state.get("targets_report") or ""))
        treatments_rep = redact_pii(str(state.get("treatments_report") or ""))
        literature_rep = redact_pii(str(state.get("literature_report") or ""))

        body_sections = [
            s
            for s in (dossier_md, targets_rep, treatments_rep, literature_rep)
            if s and not s.startswith("Not yet")
        ]
        if body_sections:
            consolidated_text = (
                f"[Consolidated Long-Term Disease Memory] {disease_name} "
                f"(EFO ID: {current_efo} | ICD-10: {icd10_str})\n"
                + "\n\n".join(body_sections[:2])
            ).strip()
            entries.append(
                MemoryEntry(
                    id=f"dossier:{current_efo}",
                    author="disease_characterisation",
                    timestamp=now_iso,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=consolidated_text)],
                    ),
                    custom_metadata={
                        "memory_type": "consolidated_disease_dossier",
                        "efo_id": current_efo,
                        "disease_name": disease_name,
                        "icd10_codes": icd10_str,
                    },
                )
            )

    return entries


class AsyncMemoryConsolidator:
    """Manages non-blocking `asyncio.create_task` background jobs for long-term memory consolidation."""

    def __init__(self) -> None:
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.completed_consolidations: int = 0

    @property
    def pending_task_count(self) -> int:
        return len(self._background_tasks)

    def schedule_consolidation(
        self,
        *,
        state_snapshot: dict[str, Any],
        app_name: str = "app",
        user_id: str = "default_user",
        session_id: str | None = None,
        memory_service: BaseMemoryService | None = None,
        callback_context: CallbackContext | None = None,
    ) -> asyncio.Task[int]:
        """Spawns an asynchronous background task (`asyncio.create_task`) to consolidate session findings."""
        task = asyncio.create_task(
            self._consolidate_worker(
                state_snapshot=state_snapshot,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_service=memory_service,
                callback_context=callback_context,
            ),
            name=f"memory_consolidation_{state_snapshot.get('efo_id', 'session')}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _consolidate_worker(
        self,
        *,
        state_snapshot: dict[str, Any],
        app_name: str,
        user_id: str,
        session_id: str | None,
        memory_service: BaseMemoryService | None,
        callback_context: CallbackContext | None,
    ) -> int:
        """Background worker that extracts, deduplicates, embeds, and persists long-term disease memories."""
        with tracer.start_as_current_span(
            "memory.async_consolidation"
        ) as span:
            efo_id = str(state_snapshot.get("efo_id") or "")
            span.set_attribute("memory.async_background", True)
            span.set_attribute("disease.efo_id", efo_id)

            entries = build_consolidated_memory_entries_from_state(
                state_snapshot
            )
            span.set_attribute("memory.entries_consolidated", len(entries))
            if not entries:
                return 0

            target_service = memory_service
            if target_service is None:
                from app.app_utils.services import get_memory_service

                target_service = get_memory_service()

            try:
                await target_service.add_memory(
                    app_name=app_name,
                    user_id=user_id,
                    memories=entries,
                    custom_metadata={
                        "session_id": session_id or "background",
                        "wait_for_completion": False,
                        "disable_consolidation": False,
                    },
                )
                if callback_context is not None:
                    try:
                        await callback_context.add_session_to_memory()
                    except Exception:
                        pass

                self.completed_consolidations += len(entries)
                logger.info(
                    "Async background memory consolidation completed for %d disease dossier(s)",
                    len(entries),
                    extra={
                        "structured_data": {
                            "event_type": "async_memory_consolidation",
                            "efo_id": efo_id,
                            "entries_consolidated": len(entries),
                            "session_id": session_id,
                        }
                    },
                )
                return len(entries)
            except Exception as exc:
                logger.warning(
                    "Async background memory consolidation failed: %s", exc
                )
                return 0

    async def drain_background_tasks(self, timeout: float = 10.0) -> None:
        """Waits for all scheduled background consolidation tasks to finish."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        await asyncio.wait(pending, timeout=timeout)


memory_consolidator = AsyncMemoryConsolidator()


async def trigger_async_memory_consolidation(
    callback_context: CallbackContext,
) -> None:
    """ADK `after_agent_callback` that schedules non-blocking background memory consolidation."""
    try:
        state_obj = getattr(callback_context, "state", None)
        state_dict = (
            state_obj.to_dict()
            if hasattr(state_obj, "to_dict")
            else dict(state_obj or {})
        )
    except Exception:
        state_dict = {}

    if not state_dict:
        return

    inv_ctx = getattr(callback_context, "_invocation_context", None)
    session = getattr(inv_ctx, "session", None)
    app_name = getattr(session, "app_name", None) or "app"
    user_id = getattr(session, "user_id", None) or "default_user"
    session_id = getattr(session, "id", None)
    mem_service = getattr(inv_ctx, "memory_service", None)

    memory_consolidator.schedule_consolidation(
        state_snapshot=state_dict,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        memory_service=mem_service,
        callback_context=callback_context,
    )


async def search_long_term_disease_memory(
    query: Annotated[
        str,
        Field(
            min_length=2,
            max_length=300,
            description=(
                "Semantic search query (disease name, EFO/MONDO ID, gene symbol, or drug name) "
                "to search across persistent long-term vector memory and prior sessions."
            ),
        ),
    ],
    top_k: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            description="Maximum number of matching long-term memory records to retrieve (1 to 10).",
        ),
    ] = 5,
    tool_context: ToolContext | None = None,
) -> SearchLongTermMemoryOutput:
    """Searches persistent long-term vector memory (and Vertex AI RAG / Memory Bank) for prior disease dossiers.

    Recovery Instructions:
        - If `status == "validation_error"`: Provide a non-empty `query` (>= 2 chars) and `top_k` between 1 and 10.
        - If `status == "empty"` (`returned_count == 0`): No prior characterisation for this query exists in
          long-term memory yet. Run `disease_characterisation_pipeline` (or `resolve_disease_ontology`) to
          characterize the disease fresh from Open Targets, ChEMBL, ClinicalTrials.gov, and Europe PMC.
        - If `status == "error"`: Fall back to fresh tool execution via `disease_characterisation_pipeline`.

    Args:
        query: Disease name, ontology ID, target symbol, or therapeutic concept to search.
        top_k: Maximum number of memory matches to return (1 to 10).

    Returns:
        Strict `SearchLongTermMemoryOutput` dictionary containing `status`, `query`,
        `returned_count`, `matches`, and `recovery_instructions`.
    """
    raw_query = redact_pii((query or "").strip())
    try:
        validated_in = SearchLongTermMemoryInput(query=raw_query, top_k=top_k)
        cleaned_query = validated_in.query
        capped_k = validated_in.top_k
    except ValidationError as val_exc:
        return validate_and_dump(  # type: ignore[return-value]
            SearchLongTermMemoryOutput,
            {
                "status": "validation_error",
                "query": raw_query,
                "returned_count": 0,
                "matches": [],
                "message": f"Invalid parameters for search_long_term_disease_memory: {val_exc.errors()[0]['msg']}",
                "recovery_instructions": (
                    "RECOVERY: Pass a non-empty `query` string (at least 2 characters) and "
                    "`top_k` between 1 and 10."
                ),
            },
        )

    try:
        from app.app_utils.services import get_memory_service

        inv_ctx = (
            getattr(tool_context, "_invocation_context", None)
            if tool_context is not None
            else None
        )
        session = getattr(inv_ctx, "session", None)
        app_name = getattr(session, "app_name", None) or "app"
        user_id = getattr(session, "user_id", None) or "default_user"

        mem_service = get_memory_service()
        matches_list: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        if isinstance(mem_service, PersistentVectorMemoryService):
            scored = await mem_service.search_with_scores(
                app_name=app_name,
                user_id=user_id,
                query=cleaned_query,
                top_k=capped_k,
            )
            for score, entry in scored:
                parts = (entry.content.parts if entry.content else None) or []
                text = "\n".join(
                    p.text for p in parts if getattr(p, "text", None)
                )
                meta = entry.custom_metadata or {}
                m_id = entry.id or meta.get("efo_id") or text[:32]
                if m_id in seen_ids:
                    continue
                seen_ids.add(str(m_id))
                matches_list.append(
                    {
                        "memory_id": entry.id,
                        "efo_id": str(meta.get("efo_id"))
                        if meta.get("efo_id")
                        else None,
                        "disease_name": str(meta.get("disease_name"))
                        if meta.get("disease_name")
                        else None,
                        "timestamp": entry.timestamp,
                        "author": entry.author,
                        "similarity_score": round(float(score), 4),
                        "content_snippet": text[:1500],
                    }
                )

        if tool_context is not None and hasattr(tool_context, "search_memory"):
            try:
                ctx_resp = await tool_context.search_memory(cleaned_query)
                for entry in getattr(ctx_resp, "memories", []) or []:
                    parts = (
                        entry.content.parts if entry.content else None
                    ) or []
                    text = "\n".join(
                        p.text for p in parts if getattr(p, "text", None)
                    )
                    meta = entry.custom_metadata or {}
                    m_id = entry.id or meta.get("efo_id") or text[:32]
                    if m_id in seen_ids or not text:
                        continue
                    seen_ids.add(str(m_id))
                    matches_list.append(
                        {
                            "memory_id": entry.id,
                            "efo_id": str(meta.get("efo_id"))
                            if meta.get("efo_id")
                            else None,
                            "disease_name": str(meta.get("disease_name"))
                            if meta.get("disease_name")
                            else None,
                            "timestamp": entry.timestamp,
                            "author": entry.author,
                            "similarity_score": meta.get("similarity_score"),
                            "content_snippet": text[:1500],
                        }
                    )
            except Exception:
                pass

        matches_list = matches_list[:capped_k]
        if not matches_list:
            return validate_and_dump(  # type: ignore[return-value]
                SearchLongTermMemoryOutput,
                {
                    "status": "empty",
                    "query": cleaned_query,
                    "returned_count": 0,
                    "matches": [],
                    "message": f"No consolidated long-term memory entries found matching '{cleaned_query}'.",
                    "recovery_instructions": (
                        f"RECOVERY: '{cleaned_query}' has not been characterized in persistent memory yet. "
                        "Call `disease_characterisation_pipeline` to run a fresh characterisation and "
                        "automatically consolidate it into long-term vector memory."
                    ),
                },
            )

        return validate_and_dump(  # type: ignore[return-value]
            SearchLongTermMemoryOutput,
            {
                "status": "success",
                "query": cleaned_query,
                "returned_count": len(matches_list),
                "matches": matches_list,
                "recovery_instructions": None,
            },
        )
    except Exception as exc:
        logger.exception(
            "Error searching long-term disease memory for '%s'", cleaned_query
        )
        return validate_and_dump(  # type: ignore[return-value]
            SearchLongTermMemoryOutput,
            {
                "status": "error",
                "query": cleaned_query,
                "returned_count": 0,
                "matches": [],
                "message": f"Long-term memory search error: {exc}",
                "recovery_instructions": (
                    "RECOVERY: Long-term vector memory lookup encountered an error. "
                    "Proceed by calling `disease_characterisation_pipeline` or the live biomedical tools."
                ),
            },
        )


def get_optional_vertex_ai_search_tools() -> list[Any]:
    """Returns configured `VertexAiSearchTool` instances when Vertex AI Search env vars are set."""
    data_store_id = os.environ.get("VERTEX_AI_SEARCH_DATA_STORE_ID")
    search_engine_id = os.environ.get("VERTEX_AI_SEARCH_ENGINE_ID")
    if not data_store_id and not search_engine_id:
        return []
    try:
        from google.adk.tools import VertexAiSearchTool

        return [
            VertexAiSearchTool(
                data_store_id=data_store_id if not search_engine_id else None,
                search_engine_id=search_engine_id,
                bypass_multi_tools_limit=True,
            )
        ]
    except Exception as exc:
        logger.warning("Could not initialize VertexAiSearchTool: %s", exc)
        return []
