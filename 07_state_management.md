# State Management Patterns for Multi-Agent Systems

> **Status**: Research & Recommendations
> **Last Updated**: 2026-02-15
> **Parent**: [00_architecture.md](00_architecture.md)
> **Scope**: Firestore, ADK sessions, conversation state, artifact storage, Vector Search, local development

---

## 1. Firestore for Agent State

### Why Firestore

Firestore is already specified as our State Store in the architecture doc (`00_architecture.md` section 8). The choice is well-suited for multi-agent systems because:

| Property | Benefit for Multi-Agent |
|----------|------------------------|
| **Real-time listeners** | Stream phase changes, vote updates, and Blueprint patches to the UI |
| **Transactions** | Atomic Blueprint patching — prevent two consultants from conflicting writes |
| **Subcollections** | Natural hierarchy: session > transcript > messages |
| **Offline support** | Firestore emulator for local dev (no cloud dependency) |
| **TTL** | Auto-expire abandoned sessions without cleanup jobs |
| **1 MB doc limit** | Forces good schema discipline — no giant monolith docs |

### Recommended Document Structure

The key design decision: **flat collections with subcollections for large/growing data**.

```
firestore/
├── sessions/{session_id}                    ← DiscussionObject (top-level state)
│   ├── transcript/{sequence_id}             ← Individual messages (subcollection)
│   ├── blueprints/{version}                 ← Blueprint versions (subcollection)
│   ├── patches/{patch_id}                   ← Blueprint patch history (subcollection)
│   ├── decisions/{decision_id}              ← Flight Recorder entries (subcollection)
│   └── phase_summaries/{phase_name}         ← State summaries per phase (subcollection)
│
├── inventory/{asset_id}                     ← Full asset metadata
│   └── versions/{version_id}               ← Asset version history (subcollection)
│
└── users/{user_id}                          ← User profiles and preferences
    └── sessions/{session_id}               ← User's session references (subcollection)
```

### Top-Level Session Document

Keep the session document **lean** — state machine control data only. Heavy data goes into subcollections.

```python
# sessions/{session_id} — the "control plane" document
# Target size: < 10 KB (well under Firestore's 1 MB limit)
{
    "session_id": "bio-project-123",
    "problem_statement": "Screen 100K compounds against KRAS G12C...",
    "created_at": "2026-02-15T10:00:00Z",
    "updated_at": "2026-02-15T10:12:00Z",
    "status": "IN_PROGRESS",  # PENDING | IN_PROGRESS | COMPLETED | ESCALATED | ABANDONED

    # Board membership — small, bounded list
    "board_members": [
        {"agent_id": "computational_chemist", "role": "domain_expert", "has_voted": False},
        {"agent_id": "budget_controller", "role": "standing_member", "has_voted": False},
    ],

    # State machine — the "control register"
    "state_machine": {
        "current_phase": "CRITIQUE",
        "phase_round": 2,
        "total_rounds": 6,
        "max_total_rounds": 15,
        "mandatory_speakers_remaining": ["compliance_officer"],
    },

    # Token budget — running counters
    "token_budget": {
        "total_used": 45000,
        "max_budget": 500000,
    },

    # Pointers to latest artifacts (not the artifacts themselves)
    "latest_blueprint_version": 3,
    "latest_phase_summary": "CRITIQUE",

    # Metrics summary (for dashboard queries)
    "metrics": {
        "objections_raised": 3,
        "objections_resolved": 2,
        "agents_proposed_new": 2,
        "agents_proposed_reuse": 1,
    }
}
```

### Why Subcollections for Transcript

The transcript is unbounded and grows with each round. If embedded in the session document:
- 15 rounds x 6 agents x ~800 tokens of content = ~72,000 tokens of text = could approach 1 MB
- Every read of the session pulls the entire transcript (wasteful for state machine checks)
- Cannot paginate or query by phase/speaker/type

With a subcollection:

```python
# sessions/{session_id}/transcript/{sequence_id}
{
    "sequence_id": 4,
    "timestamp": "2026-02-15T10:01:20Z",
    "phase": "CRITIQUE",
    "phase_round": 1,
    "speaker": "budget_controller",
    "recipient": "manager",
    "message_type": "OBJECTION",
    "content": "A100 GPUs for MM-GBSA on 1K compounds is overkill...",
    "context_scope": "SCOPED",
    "tokens_used": 650,
    "citations": ["skills/gcp/cost_optimization"],
    "meta": {
        "cost_original": "$734",
        "cost_proposed": "$180",
    }
}
```

Benefits:
- Query by phase: `where("phase", "==", "CRITIQUE")`
- Query by speaker: `where("speaker", "==", "budget_controller")`
- Query by type: `where("message_type", "==", "OBJECTION")`
- Paginate: `order_by("sequence_id").limit(10)`
- Individual messages are small (~1-3 KB each)

### Transactions for Blueprint Patching

The Dual-Stream State design (Discussion Protocol section 12) requires **atomic Blueprint updates**. Two consultants might try to patch the Blueprint simultaneously.

```python
from google.cloud import firestore

db = firestore.AsyncClient()

async def patch_blueprint(
    session_id: str,
    path: str,
    value: any,
    reason: str,
    author: str,
    previous_value: any = None,
) -> dict:
    """Atomic Blueprint patch using Firestore transaction.

    Ensures:
    1. Blueprint version is incremented atomically
    2. Patch is recorded in history
    3. Conflict detection if previous_value doesn't match current
    """
    session_ref = db.collection("sessions").document(session_id)

    @firestore.async_transactional
    async def update_in_transaction(transaction):
        # Read current state
        session_doc = await session_ref.get(transaction=transaction)
        current_version = session_doc.get("latest_blueprint_version")

        # Get current blueprint
        blueprint_ref = session_ref.collection("blueprints").document(str(current_version))
        blueprint_doc = await blueprint_ref.get(transaction=transaction)
        blueprint = blueprint_doc.to_dict()

        # Conflict detection (optimistic concurrency)
        current_at_path = _get_nested(blueprint, path)
        if previous_value is not None and current_at_path != previous_value:
            raise ConflictError(
                f"Conflict at '{path}': expected '{previous_value}', "
                f"found '{current_at_path}'"
            )

        # Apply patch to blueprint
        new_blueprint = _set_nested(blueprint, path, value)
        new_version = current_version + 1

        # Write new blueprint version
        new_blueprint_ref = session_ref.collection("blueprints").document(str(new_version))
        transaction.set(new_blueprint_ref, {
            **new_blueprint,
            "blueprint_version": new_version,
            "last_modified": firestore.SERVER_TIMESTAMP,
        })

        # Record patch
        patch_ref = session_ref.collection("patches").document()
        transaction.set(patch_ref, {
            "patch_id": new_version,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "author": author,
            "path": path,
            "old_value": current_at_path,
            "new_value": value,
            "reason": reason,
        })

        # Update session pointer
        transaction.update(session_ref, {
            "latest_blueprint_version": new_version,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        return {"version": new_version, "path": path, "value": value}

    transaction = db.transaction()
    return await update_in_transaction(transaction)
```

### Real-Time Listeners for UI

When we build the web UI (Phase 2), Firestore real-time listeners enable live discussion streaming:

```python
# Server-side: stream session updates to the UI via SSE or WebSocket
def watch_session(session_id: str):
    """Real-time listener for session state changes."""

    session_ref = db.collection("sessions").document(session_id)

    # Watch the session document for phase transitions
    def on_session_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            emit_to_client("session_update", {
                "phase": data["state_machine"]["current_phase"],
                "round": data["state_machine"]["phase_round"],
                "token_usage": data["token_budget"]["total_used"],
            })

    session_ref.on_snapshot(on_session_snapshot)

    # Watch transcript subcollection for new messages
    transcript_ref = (
        session_ref.collection("transcript")
        .order_by("sequence_id")
    )

    def on_transcript_snapshot(doc_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == "ADDED":
                msg = change.document.to_dict()
                emit_to_client("new_message", {
                    "speaker": msg["speaker"],
                    "message_type": msg["message_type"],
                    "content": msg["content"],
                    "phase": msg["phase"],
                })

    transcript_ref.on_snapshot(on_transcript_snapshot)
```

### Firestore Best Practices for This System

| Practice | Application |
|----------|-------------|
| **Avoid hotspots** | Session IDs are UUIDs (uniform distribution). Do not use monotonic IDs. |
| **Shallow documents** | Session doc is control data only (~10 KB). Transcript, blueprints, patches are subcollections. |
| **Compound indexes** | Create for `transcript` queries: `(phase, message_type)`, `(speaker, phase)`, `(phase, sequence_id)`. |
| **TTL for abandoned sessions** | Set `expireAt` field on sessions with `status == "ABANDONED"`. Firestore TTL auto-deletes. |
| **Batch writes** | When PROPOSAL phase runs parallel (4 consultants respond), batch-write all 4 transcript entries. |
| **Security rules** | In production: users can only read their own sessions. Agents authenticate via service accounts. |
| **Document size discipline** | If any field grows unbounded (transcript, patches), it must be a subcollection. |

---

## 2. ADK Session State

### How Google ADK Manages State

Google ADK (Agent Development Kit) provides a **session abstraction** for managing state across agent turns. The key concepts:

| Concept | Description |
|---------|-------------|
| **Session** | A conversation container. Has an ID, user ID, and a state dictionary. |
| **SessionService** | Interface for creating, getting, and listing sessions. Multiple implementations available. |
| **InMemorySessionService** | Default. Stores sessions in a Python dict. Lost on process restart. |
| **DatabaseSessionService** | Persists to a database. ADK provides a Firestore-backed implementation. |
| **State scopes** | `state` (session-level), `app_state` (shared across sessions), `user_state` (per-user across sessions). |

### ADK State Scopes

ADK distinguishes three state scopes, which map well to our system:

```python
# In ADK, state is accessed via the session object:

# Session state — per-discussion, per-session
# Maps to: Our DiscussionObject (phase, round, board_members, etc.)
session.state["current_phase"] = "CRITIQUE"
session.state["total_rounds"] = 6
session.state["board_members"] = [...]

# App state — shared across ALL sessions
# Maps to: Our Inventory metadata, system configuration
session.app_state["inventory_version"] = "2026-02-15"
session.app_state["max_token_budget"] = 500_000

# User state — per-user, across all their sessions
# Maps to: User preferences, session history, default consultants
session.user_state["preferred_consultants"] = ["computational_chemist"]
session.user_state["cost_threshold"] = 500
```

### Plugging in Firestore

ADK provides `DatabaseSessionService` which can be backed by Firestore. The configuration:

```python
from google.adk.sessions import DatabaseSessionService

# Option A: Use ADK's built-in database session service
# This uses ADK's own Firestore schema (automatic)
session_service = DatabaseSessionService(
    database_url="firestore://project-id/adk-sessions"
)

# Option B: Use InMemorySessionService for MVP (Phase 0-1)
from google.adk.sessions import InMemorySessionService
session_service = InMemorySessionService()
```

### Our Recommendation: Hybrid Approach

ADK's built-in session state is **too simple** for our needs. The `state` dict is a flat key-value store — it does not support subcollections, transactions, or the rich schema we need for the DiscussionObject.

**Recommended approach**: Use ADK's session service for the agent runtime, but manage our domain state (DiscussionObject, Blueprint, FlightRecorder) in our own Firestore schema.

```python
from google.adk.sessions import InMemorySessionService  # Phase 0-1
from google.adk.agents import LlmAgent
from src.state.discussion_store import DiscussionStore   # Our custom Firestore layer

# ADK manages the agent conversation session
session_service = InMemorySessionService()

# Our DiscussionStore manages the domain-specific state
discussion_store = DiscussionStore(
    firestore_client=firestore.AsyncClient(),
)

# The Manager agent's tools bridge both layers
class ManagerTools:
    def __init__(self, session_service, discussion_store):
        self.session_service = session_service
        self.discussion_store = discussion_store

    async def call_consultant(
        self, recipient: str, instruction: str, **kwargs
    ) -> str:
        # 1. Read discussion state from our Firestore
        discussion = await self.discussion_store.get_session(self.session_id)

        # 2. Build context using our ContextBuilder
        context = self.context_builder.build(discussion, kwargs.get("context_scope"))

        # 3. Call the consultant via ADK sub-agent
        response = await self.sub_agents[recipient].run(
            instruction=instruction,
            context=context,
        )

        # 4. Write transcript entry to our Firestore
        await self.discussion_store.add_transcript_entry(
            session_id=self.session_id,
            entry={
                "speaker": recipient,
                "content": response.text,
                "phase": discussion["state_machine"]["current_phase"],
                # ... other fields
            }
        )

        return response.text
```

### State Key Constants (From research-agent Pattern)

The `research-agent` reference repo uses a `state_keys.py` pattern to prevent typos. We should adopt this:

```python
# src/state/state_keys.py
"""Centralized state key constants.

Prevents typos and provides a single source of truth for all state keys
used across agents. Follows the pattern from the research-agent reference repo.
"""

# Session state keys (per-discussion)
class SessionKeys:
    CURRENT_PHASE = "current_phase"
    PHASE_ROUND = "phase_round"
    TOTAL_ROUNDS = "total_rounds"
    BOARD_MEMBERS = "board_members"
    MANDATORY_SPEAKERS_REMAINING = "mandatory_speakers_remaining"
    LATEST_BLUEPRINT_VERSION = "latest_blueprint_version"
    TOKEN_BUDGET_USED = "token_budget_used"
    TOKEN_BUDGET_MAX = "token_budget_max"
    STATUS = "status"

# App state keys (shared)
class AppKeys:
    INVENTORY_VERSION = "inventory_version"
    MAX_TOKEN_BUDGET = "max_token_budget"
    CONSULTANT_CONFIGS = "consultant_configs"

# User state keys (per-user)
class UserKeys:
    PREFERRED_CONSULTANTS = "preferred_consultants"
    COST_THRESHOLD = "cost_threshold"
    SESSION_HISTORY = "session_history"
```

### MVP vs Production State Strategy

| Phase | Session Service | Domain State | Rationale |
|-------|----------------|--------------|-----------|
| **Phase 0-1 (MVP)** | `InMemorySessionService` | In-memory Python dicts | Fast iteration, no cloud dependency |
| **Phase 2-3** | `InMemorySessionService` | Firestore (our schema) | Need persistence for async Board sessions |
| **Phase 4+** | `DatabaseSessionService` | Firestore (our schema) | Full ADK integration, production-grade |

---

## 3. Multi-Agent Conversation State

### The Three Types of State

Our system has three distinct categories of conversation state, each with different storage requirements:

| State Type | Examples | Mutability | Size | Storage |
|------------|----------|------------|------|---------|
| **Transcript** | Discussion messages, proposals, objections | Append-only | Grows linearly (50-100 entries per session) | Firestore subcollection |
| **Decision Log** | Flight Recorder entries, patch history | Append-only | Grows linearly (10-30 entries per session) | Firestore subcollection |
| **Evolving Document** | Blueprint artifact, phase summaries | Mutable (versioned) | Fixed size, multiple versions (~5 KB each) | Firestore subcollection (versioned) |

### Pattern 1: Append-Only Transcript

The discussion transcript is append-only. Messages are never edited or deleted.

```python
class TranscriptStore:
    """Manages the append-only discussion transcript."""

    def __init__(self, db: firestore.AsyncClient):
        self.db = db

    async def add_entry(self, session_id: str, entry: dict) -> str:
        """Append a new transcript entry. Returns the sequence_id."""
        session_ref = self.db.collection("sessions").document(session_id)

        # Use a transaction to atomically get next sequence_id and write
        @firestore.async_transactional
        async def _add(transaction):
            session = await session_ref.get(transaction=transaction)
            next_seq = session.get("_transcript_counter", 0) + 1

            entry_ref = session_ref.collection("transcript").document(str(next_seq))
            transaction.set(entry_ref, {
                **entry,
                "sequence_id": next_seq,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })
            transaction.update(session_ref, {
                "_transcript_counter": next_seq,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            return str(next_seq)

        transaction = self.db.transaction()
        return await _add(transaction)

    async def query_by_phase(
        self, session_id: str, phase: str
    ) -> list[dict]:
        """Get all transcript entries for a specific phase."""
        ref = (
            self.db.collection("sessions")
            .document(session_id)
            .collection("transcript")
            .where("phase", "==", phase)
            .order_by("sequence_id")
        )
        docs = await ref.get()
        return [doc.to_dict() for doc in docs]

    async def query_by_speaker(
        self, session_id: str, speaker: str
    ) -> list[dict]:
        """Get all messages from a specific speaker."""
        ref = (
            self.db.collection("sessions")
            .document(session_id)
            .collection("transcript")
            .where("speaker", "==", speaker)
            .order_by("sequence_id")
        )
        docs = await ref.get()
        return [doc.to_dict() for doc in docs]

    async def get_scoped_context(
        self, session_id: str, sequence_ids: list[int]
    ) -> list[dict]:
        """Get specific messages by sequence_id for SCOPED context."""
        # Firestore 'in' queries support up to 30 values
        ref = (
            self.db.collection("sessions")
            .document(session_id)
            .collection("transcript")
            .where("sequence_id", "in", sequence_ids)
            .order_by("sequence_id")
        )
        docs = await ref.get()
        return [doc.to_dict() for doc in docs]
```

### Pattern 2: Versioned Blueprint (Evolving Document)

The Blueprint is mutable but must maintain full version history for auditability.

```python
class BlueprintStore:
    """Manages versioned Blueprint artifacts with atomic patching."""

    def __init__(self, db: firestore.AsyncClient):
        self.db = db

    async def get_latest(self, session_id: str) -> dict:
        """Get the latest Blueprint version."""
        session = await (
            self.db.collection("sessions")
            .document(session_id)
            .get()
        )
        version = session.get("latest_blueprint_version", 0)
        if version == 0:
            return None

        doc = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("blueprints")
            .document(str(version))
            .get()
        )
        return doc.to_dict()

    async def get_version(self, session_id: str, version: int) -> dict:
        """Get a specific Blueprint version (for diff/audit)."""
        doc = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("blueprints")
            .document(str(version))
            .get()
        )
        return doc.to_dict()

    async def get_patch_history(self, session_id: str) -> list[dict]:
        """Get all patches applied to the Blueprint, in order."""
        docs = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("patches")
            .order_by("timestamp")
            .get()
        )
        return [doc.to_dict() for doc in docs]
```

### Pattern 3: Phase Summaries (Rolling Context)

Phase summaries serve as compressed context for the Manager's token budget management.

```python
class PhaseSummaryStore:
    """Manages rolling phase summaries for token-efficient context."""

    def __init__(self, db: firestore.AsyncClient):
        self.db = db

    async def save_summary(
        self, session_id: str, phase: str, summary: dict
    ) -> None:
        """Save or update a phase summary."""
        await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("phase_summaries")
            .document(phase)
            .set(summary)
        )

    async def get_all_summaries(self, session_id: str) -> dict:
        """Get all phase summaries (for SUMMARY context scope)."""
        docs = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("phase_summaries")
            .order_by("__name__")  # Phase names are ordered: INIT < PROPOSAL < CRITIQUE...
            .get()
        )
        return {doc.id: doc.to_dict() for doc in docs}

    async def build_context_summary(self, session_id: str) -> str:
        """Build a compact text summary for LLM context injection.

        This is what gets sent to consultants with context_scope=SUMMARY.
        Target: < 2,000 tokens for all phases combined.
        """
        summaries = await self.get_all_summaries(session_id)
        parts = []
        for phase, data in summaries.items():
            parts.append(f"[{phase}]: {data.get('summary', 'No summary')}")
            if data.get("unresolved_issues"):
                parts.append(f"  Unresolved: {', '.join(data['unresolved_issues'])}")
        return "\n".join(parts)
```

### Unified Discussion Store

A facade that combines all three patterns:

```python
class DiscussionStore:
    """Unified access to all discussion state.

    This is the primary interface for the Manager agent's tools.
    Wraps TranscriptStore, BlueprintStore, PhaseSummaryStore, and
    the session control document.
    """

    def __init__(self, db: firestore.AsyncClient):
        self.db = db
        self.transcript = TranscriptStore(db)
        self.blueprint = BlueprintStore(db)
        self.summaries = PhaseSummaryStore(db)

    async def create_session(self, session_id: str, problem_statement: str, board_members: list) -> dict:
        """Initialize a new discussion session."""
        session_data = {
            "session_id": session_id,
            "problem_statement": problem_statement,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "status": "IN_PROGRESS",
            "board_members": board_members,
            "state_machine": {
                "current_phase": "INIT",
                "phase_round": 0,
                "total_rounds": 0,
                "max_total_rounds": 15,
                "mandatory_speakers_remaining": [],
            },
            "token_budget": {
                "total_used": 0,
                "max_budget": 500_000,
            },
            "latest_blueprint_version": 0,
            "_transcript_counter": 0,
        }
        await self.db.collection("sessions").document(session_id).set(session_data)
        return session_data

    async def advance_phase(self, session_id: str, new_phase: str, mandatory_speakers: list = None) -> None:
        """Advance the state machine to the next phase."""
        ref = self.db.collection("sessions").document(session_id)

        @firestore.async_transactional
        async def _advance(transaction):
            doc = await ref.get(transaction=transaction)
            sm = doc.get("state_machine")
            transaction.update(ref, {
                "state_machine.current_phase": new_phase,
                "state_machine.phase_round": 0,
                "state_machine.total_rounds": sm["total_rounds"] + 1,
                "state_machine.mandatory_speakers_remaining": mandatory_speakers or [],
                "updated_at": firestore.SERVER_TIMESTAMP,
            })

        transaction = self.db.transaction()
        await _advance(transaction)

    async def get_session(self, session_id: str) -> dict:
        """Get the session control document."""
        doc = await self.db.collection("sessions").document(session_id).get()
        return doc.to_dict()
```

---

## 4. Blueprint/Artifact Storage: Firestore vs GCS vs Both

### The Decision Framework

| Factor | Firestore | GCS | Recommendation |
|--------|-----------|-----|----------------|
| **Live Blueprint** (during discussion) | Transactions, real-time updates, versioning | No transactions, no real-time listeners | **Firestore** |
| **Finalized Blueprint** (post-approval) | Already there from discussion | Better for long-term archival, integrates with CI/CD | **Both** (copy to GCS on finalize) |
| **Large artifacts** (>1 MB) | 1 MB document limit | No size limit | **GCS** |
| **Binary artifacts** (Docker images, model weights) | Not suitable | Designed for this | **GCS** |
| **Queryable metadata** | Native queries, indexes | No query capability | **Firestore** |
| **Cost at scale** | $0.06/100K reads | $0.004/10K reads | **GCS cheaper** for read-heavy archival |

### Our Recommendation: Tiered Storage

```
                    ┌────────────────────────────┐
                    │   During Discussion         │
                    │   Firestore (live editing)  │
                    │   - Blueprint v1, v2, v3... │
                    │   - Transactions for patches│
                    │   - Real-time for UI        │
                    └──────────┬─────────────────┘
                               │ FINALIZE phase
                               ▼
                    ┌────────────────────────────┐
                    │   On Finalization           │
                    │   Copy to GCS              │
                    │   gs://blueprints/          │
                    │     {session_id}/           │
                    │       blueprint.yaml        │
                    │       flight_record.json    │
                    │       transcript.jsonl      │
                    └──────────┬─────────────────┘
                               │ Dev Team reads from GCS
                               ▼
                    ┌────────────────────────────┐
                    │   Dev Team Consumes         │
                    │   Reads blueprint.yaml      │
                    │   from GCS (immutable)      │
                    └────────────────────────────┘
```

### Implementation

```python
from google.cloud import storage
import yaml

class ArtifactStore:
    """Manages the lifecycle of Blueprint artifacts across Firestore and GCS."""

    def __init__(self, db: firestore.AsyncClient, gcs_bucket: str):
        self.db = db
        self.gcs = storage.Client()
        self.bucket = self.gcs.bucket(gcs_bucket)

    async def finalize_blueprint(self, session_id: str) -> str:
        """Called in FINALIZE phase. Copies finalized Blueprint to GCS.

        Returns the GCS URI of the finalized Blueprint.
        """
        # 1. Get latest Blueprint from Firestore
        session = await self.db.collection("sessions").document(session_id).get()
        version = session.get("latest_blueprint_version")
        blueprint = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("blueprints")
            .document(str(version))
            .get()
        )

        # 2. Convert to YAML and upload to GCS
        blueprint_yaml = yaml.dump(blueprint.to_dict(), default_flow_style=False)
        gcs_path = f"blueprints/{session_id}/blueprint.yaml"
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_string(blueprint_yaml, content_type="application/x-yaml")

        # 3. Export full transcript as JSONL
        transcript_docs = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("transcript")
            .order_by("sequence_id")
            .get()
        )
        transcript_jsonl = "\n".join(
            json.dumps(doc.to_dict(), default=str) for doc in transcript_docs
        )
        transcript_blob = self.bucket.blob(f"blueprints/{session_id}/transcript.jsonl")
        transcript_blob.upload_from_string(transcript_jsonl, content_type="application/jsonl")

        # 4. Export Flight Recorder
        decisions = await (
            self.db.collection("sessions")
            .document(session_id)
            .collection("decisions")
            .order_by("timestamp")
            .get()
        )
        flight_record = {
            "session_id": session_id,
            "decision_tree": [doc.to_dict() for doc in decisions],
        }
        fr_blob = self.bucket.blob(f"blueprints/{session_id}/flight_record.json")
        fr_blob.upload_from_string(
            json.dumps(flight_record, default=str, indent=2),
            content_type="application/json"
        )

        # 5. Update session with GCS pointer
        await self.db.collection("sessions").document(session_id).update({
            "status": "COMPLETED",
            "gcs_blueprint_uri": f"gs://{self.bucket.name}/{gcs_path}",
            "completed_at": firestore.SERVER_TIMESTAMP,
        })

        return f"gs://{self.bucket.name}/{gcs_path}"
```

### YAML vs JSON for Blueprints

| Format | Pros | Cons | Verdict |
|--------|------|------|---------|
| **YAML** | Human-readable, comments, multi-line strings, matches our schema spec | Slower to parse, indentation-sensitive | **GCS archival format** |
| **JSON** | Fast parsing, Firestore-native, no ambiguity | Less readable, no comments | **Firestore working format** |

Use JSON inside Firestore (it is native). Convert to YAML when exporting to GCS for the Dev Team and human review.

---

## 5. Vector Search for Semantic Inventory

### How Vertex AI Vector Search Works

Vertex AI Vector Search (formerly Matching Engine) is Google Cloud's managed service for approximate nearest neighbor (ANN) search.

#### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Embedding     │     │ Vector Search │     │ Query Time       │
│ Generation    │     │ Index         │     │                  │
│               │     │               │     │                  │
│ text-embed-   │────▶│ ScaNN-based   │────▶│ k-NN query       │
│ ding-005      │     │ ANN index     │     │ returns top-K    │
│               │     │ (managed)     │     │ with scores      │
└──────────────┘     └──────────────┘     └──────────────────┘
```

#### Key Operations

**1. Index Creation**

```python
from google.cloud import aiplatform

# Create the index (one-time setup)
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="inventory-index",
    description="Semantic inventory for BioTech agent assets",
    dimensions=768,  # text-embedding-005 dimension
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    shard_size="SHARD_SIZE_SMALL",  # For < 10K vectors
    # For our scale (hundreds of assets), SMALL shard is sufficient
)

# Deploy the index to an endpoint
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="inventory-endpoint",
    public_endpoint_enabled=True,  # For simplicity; use VPC for production
)

deployed_index = index_endpoint.deploy_index(
    index=index,
    deployed_index_id="inventory-deployed",
    machine_type="e2-standard-2",  # Minimum for small indexes
    min_replica_count=1,
    max_replica_count=1,  # Scale up later
)
```

**2. Upsert (Add/Update Vectors)**

```python
from google.cloud import aiplatform_v1

async def upsert_asset(asset_id: str, embedding: list[float], metadata: dict):
    """Add or update an asset in the Vector Search index."""
    # Vector Search uses a specific format for upserts
    datapoint = aiplatform_v1.IndexDatapoint(
        datapoint_id=asset_id,
        feature_vector=embedding,
        restricts=[
            # Restricts enable pre-filtering (like WHERE clauses)
            aiplatform_v1.IndexDatapoint.Restriction(
                namespace="category",
                allow_list=[metadata["category"]],  # "SKILL", "AGENT", etc.
            ),
            aiplatform_v1.IndexDatapoint.Restriction(
                namespace="status",
                allow_list=[metadata["status"]],  # "ACTIVE", "DEPRECATED"
            ),
        ],
    )

    # Upsert to the index
    index.upsert_datapoints(datapoints=[datapoint])
```

**3. Querying**

```python
async def search_inventory(
    query: str,
    category: str = None,
    status: str = "ACTIVE",
    top_k: int = 10,
    min_score: float = 0.65,
) -> list[dict]:
    """Semantic search over the inventory."""

    # 1. Generate query embedding
    embedding = generate_embedding(query)

    # 2. Build restriction filters
    restricts = []
    if status:
        restricts.append(
            aiplatform_v1.IndexDatapoint.Restriction(
                namespace="status",
                allow_list=[status],
            )
        )
    if category:
        restricts.append(
            aiplatform_v1.IndexDatapoint.Restriction(
                namespace="category",
                allow_list=[category],
            )
        )

    # 3. Query the index
    response = index_endpoint.find_neighbors(
        deployed_index_id="inventory-deployed",
        queries=[embedding],
        num_neighbors=top_k,
        restricts=restricts,
    )

    # 4. Filter by minimum score and enrich with Firestore metadata
    results = []
    for neighbor in response[0]:
        if neighbor.distance >= min_score:
            # Fetch full metadata from Firestore
            metadata = await get_asset_from_firestore(neighbor.datapoint_id)
            results.append({
                "asset_id": neighbor.datapoint_id,
                "score": neighbor.distance,
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "category": metadata.get("category"),
                "quick_start": metadata.get("quick_start"),
            })

    return results
```

#### Pricing (as of early 2026)

| Component | Cost | Notes |
|-----------|------|-------|
| **Index storage** | ~$0.06/GB/month | For our scale (<1 GB), negligible |
| **Deployed index** | ~$0.10-0.50/hour per replica | The main cost. Minimum 1 replica. |
| **Queries** | Included with deployed index | No per-query cost |
| **Embedding generation** | ~$0.025/1M characters | Using text-embedding-005 |

For our scale (hundreds of assets, low QPS):
- Estimated monthly cost: **$75-$370/month** for a single small replica
- The deployed index is the major cost — it runs 24/7

#### The Progressive Strategy: In-Memory to Vector Search

We should NOT deploy Vector Search in Phase 0-1. The progression:

```
Phase 0-1 (MVP):     In-memory search (Python list + cosine similarity)
                      Cost: $0/month
                      Latency: <10ms
                      Scale: Hundreds of assets (fine)

Phase 2-3:            Firestore with embedding field + client-side similarity
                      Cost: Firestore pricing only
                      Latency: ~100ms
                      Scale: Thousands of assets

Phase 4+ (Production): Vertex AI Vector Search
                        Cost: ~$75-370/month
                        Latency: <50ms at scale
                        Scale: Millions of assets
```

### In-Memory MVP Implementation

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class InventoryAsset:
    asset_id: str
    name: str
    description: str
    category: str
    status: str
    tags: list[str]
    quick_start: str
    cost_hint: str
    embedding: list[float]
    metadata_pointer: str  # Firestore path for full metadata

class InMemoryInventory:
    """MVP inventory search using in-memory cosine similarity.

    Good enough for Phase 0-1 with hundreds of assets.
    Drop-in replaceable with Vertex AI Vector Search later.
    """

    def __init__(self):
        self.assets: dict[str, InventoryAsset] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._asset_ids: list[str] = []
        self._dirty = True

    def upsert(self, asset: InventoryAsset) -> None:
        """Add or update an asset."""
        self.assets[asset.asset_id] = asset
        self._dirty = True

    def _rebuild_index(self) -> None:
        """Rebuild the numpy matrix for fast similarity computation."""
        if not self._dirty:
            return
        self._asset_ids = list(self.assets.keys())
        embeddings = [self.assets[aid].embedding for aid in self._asset_ids]
        if embeddings:
            self._embedding_matrix = np.array(embeddings)
            # Normalize for cosine similarity via dot product
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._embedding_matrix = self._embedding_matrix / norms
        self._dirty = False

    def search(
        self,
        query_embedding: list[float],
        category: str = None,
        status: str = "ACTIVE",
        top_k: int = 10,
        min_score: float = 0.65,
    ) -> list[dict]:
        """Search for assets by semantic similarity.

        Interface matches the production search_inventory() tool.
        """
        self._rebuild_index()
        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        # Normalize query
        query = np.array(query_embedding)
        query = query / np.linalg.norm(query)

        # Compute cosine similarities (dot product of normalized vectors)
        similarities = self._embedding_matrix @ query

        # Sort by similarity (descending)
        indices = np.argsort(similarities)[::-1]

        results = []
        for idx in indices:
            if len(results) >= top_k:
                break

            asset_id = self._asset_ids[idx]
            asset = self.assets[asset_id]
            score = float(similarities[idx])

            # Apply filters
            if score < min_score:
                break  # Sorted descending, so all remaining are below threshold
            if category and asset.category != category:
                continue
            if status and asset.status != status:
                continue

            results.append({
                "asset_id": asset.asset_id,
                "score": score,
                "name": asset.name,
                "description": asset.description,
                "category": asset.category,
                "quick_start": asset.quick_start,
            })

        return results

    def load_from_json(self, path: str) -> int:
        """Load pre-computed assets from a JSON file (bootstrap)."""
        import json
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self.upsert(InventoryAsset(**item))
        return len(data)
```

### Abstraction Layer for Progressive Upgrade

```python
from abc import ABC, abstractmethod

class InventorySearchBackend(ABC):
    """Abstract interface for inventory search.

    Implementations:
    - InMemoryInventory (Phase 0-1)
    - FirestoreInventory (Phase 2-3, with embeddings stored in docs)
    - VertexVectorSearch (Phase 4+)
    """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        category: str = None,
        status: str = "ACTIVE",
        top_k: int = 10,
        min_score: float = 0.65,
    ) -> list[dict]:
        ...

    @abstractmethod
    async def upsert(self, asset_id: str, embedding: list[float], metadata: dict) -> None:
        ...

    @abstractmethod
    async def delete(self, asset_id: str) -> None:
        ...


class InventoryService:
    """The tool interface exposed to agents.

    Wraps the search backend and adds Firestore metadata enrichment.
    """

    def __init__(
        self,
        search_backend: InventorySearchBackend,
        metadata_store: firestore.AsyncClient,
        embedding_model: str = "text-embedding-005",
    ):
        self.search_backend = search_backend
        self.metadata_store = metadata_store
        self.embedding_model = embedding_model

    async def search_inventory(
        self,
        query: str,
        category: str = None,
        status: str = "ACTIVE",
        top_k: int = 10,
        min_score: float = 0.65,
    ) -> list[dict]:
        """The search_inventory() tool used by the Manager agent."""
        # 1. Generate embedding for the query
        embedding = await self._embed(query)

        # 2. Search (delegates to whichever backend is configured)
        results = await self.search_backend.search(
            query_embedding=embedding,
            category=category,
            status=status,
            top_k=top_k,
            min_score=min_score,
        )

        return results

    async def get_asset_details(self, asset_id: str) -> dict:
        """Fetch full metadata from Firestore."""
        doc = await self.metadata_store.collection("inventory").document(asset_id).get()
        return doc.to_dict()
```

---

## 6. Local Development

### What Can Be Emulated Locally

| Service | Local Option | Fidelity | Setup |
|---------|-------------|----------|-------|
| **Firestore** | Firebase Emulator Suite | High (full API compatibility) | `firebase emulators:start --only firestore` |
| **Vector Search** | In-memory (numpy) | Medium (same interface, different perf) | Our `InMemoryInventory` class |
| **GCS** | Local filesystem or `fake-gcs-server` | High | Docker container |
| **Pub/Sub** | Pub/Sub emulator | High | `gcloud beta emulators pubsub start` |
| **BigQuery** | No emulator (use sandbox) | N/A | Must use cloud or skip |
| **Vertex AI (LLM)** | No emulator (use API) | N/A | Must call Vertex AI API |
| **Embeddings** | Local model (e5-small) or cache | Medium-Low | `sentence-transformers` for offline |

### Firestore Emulator Setup

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize (one-time, in project root)
firebase init emulators
# Select: Firestore Emulator
# Port: 8080 (default)

# Start emulator
firebase emulators:start --only firestore

# The emulator provides:
# - Full Firestore API at localhost:8080
# - Emulator UI at localhost:4000 (inspect data visually)
# - Data export/import for repeatable tests
```

### Connecting to the Emulator in Code

```python
import os
from google.cloud import firestore

def get_firestore_client() -> firestore.AsyncClient:
    """Returns a Firestore client configured for local or cloud.

    Set FIRESTORE_EMULATOR_HOST=localhost:8080 for local development.
    The Firestore SDK auto-detects this env var.
    """
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        # Local emulator — no credentials needed
        return firestore.AsyncClient(
            project="demo-project",  # Any string works with emulator
        )
    else:
        # Production — uses default credentials
        return firestore.AsyncClient()
```

### Complete Local Dev Stack

```yaml
# docker-compose.dev.yaml
version: "3.8"

services:
  # Firestore emulator (primary state store)
  firestore:
    image: google/cloud-sdk:latest
    command: >
      gcloud emulators firestore start
      --host-port=0.0.0.0:8080
      --project=demo-project
    ports:
      - "8080:8080"

  # Fake GCS server (for Blueprint archival)
  gcs:
    image: fsouza/fake-gcs-server:latest
    command: -scheme http -port 4443
    ports:
      - "4443:4443"
    volumes:
      - gcs-data:/data

  # Pub/Sub emulator (for async messaging, Phase 2+)
  pubsub:
    image: google/cloud-sdk:latest
    command: >
      gcloud beta emulators pubsub start
      --host-port=0.0.0.0:8085
      --project=demo-project
    ports:
      - "8085:8085"

volumes:
  gcs-data:
```

### Local Environment Configuration

```python
# src/config.py
"""Environment-aware configuration for local and cloud deployment."""

import os
from dataclasses import dataclass, field

@dataclass
class Config:
    """Application configuration with sensible defaults for local dev."""

    # Environment
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "local")
    )

    # Firestore
    firestore_project: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT", "demo-project")
    )

    # Inventory search backend
    inventory_backend: str = field(
        default_factory=lambda: os.getenv("INVENTORY_BACKEND", "in_memory")
    )
    # "in_memory" for Phase 0-1
    # "firestore_embeddings" for Phase 2-3
    # "vertex_vector_search" for Phase 4+

    # Embedding model
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    )
    use_local_embeddings: bool = field(
        default_factory=lambda: os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    )

    # GCS
    blueprint_bucket: str = field(
        default_factory=lambda: os.getenv("BLUEPRINT_BUCKET", "biotech-blueprints")
    )

    # ADK
    adk_session_backend: str = field(
        default_factory=lambda: os.getenv("ADK_SESSION_BACKEND", "in_memory")
    )
    # "in_memory" for Phase 0-1
    # "firestore" for Phase 4+

    # LLM
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gemini-3-pro-preview")
    )
    linting_model: str = field(
        default_factory=lambda: os.getenv("LINTING_MODEL", "gemini-3-flash-preview")
    )

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def create_services(config: Config):
    """Factory that wires up services based on configuration.

    Ensures the same code runs locally and in production —
    only the backends change.
    """
    from google.cloud import firestore as fs

    # Firestore (emulator or cloud — SDK auto-detects)
    db = fs.AsyncClient(project=config.firestore_project)

    # Discussion store (always Firestore-backed)
    discussion_store = DiscussionStore(db)

    # Inventory search backend (progressive)
    if config.inventory_backend == "in_memory":
        from src.inventory.in_memory import InMemoryInventory
        search_backend = InMemoryInventory()
    elif config.inventory_backend == "vertex_vector_search":
        from src.inventory.vertex_search import VertexVectorSearch
        search_backend = VertexVectorSearch(
            index_endpoint=os.getenv("VECTOR_SEARCH_ENDPOINT"),
            deployed_index_id=os.getenv("DEPLOYED_INDEX_ID"),
        )
    else:
        raise ValueError(f"Unknown inventory backend: {config.inventory_backend}")

    # Inventory service (wraps backend + metadata store)
    inventory_service = InventoryService(
        search_backend=search_backend,
        metadata_store=db,
        embedding_model=config.embedding_model,
    )

    # Artifact store (Firestore + GCS)
    artifact_store = ArtifactStore(db=db, gcs_bucket=config.blueprint_bucket)

    # ADK session service
    if config.adk_session_backend == "in_memory":
        from google.adk.sessions import InMemorySessionService
        session_service = InMemorySessionService()
    else:
        from google.adk.sessions import DatabaseSessionService
        session_service = DatabaseSessionService(
            database_url=f"firestore://{config.firestore_project}/adk-sessions"
        )

    return {
        "db": db,
        "discussion_store": discussion_store,
        "inventory_service": inventory_service,
        "artifact_store": artifact_store,
        "session_service": session_service,
    }
```

### Embedding Cache for Offline Development

For local development without calling Vertex AI for embeddings:

```python
import json
import hashlib
from pathlib import Path

class EmbeddingCache:
    """Cache embeddings locally to avoid repeated API calls during development.

    First call to embed() hits the API and caches the result.
    Subsequent calls return the cached embedding.
    """

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def embed(self, text: str, model: str = "text-embedding-005") -> list[float]:
        """Get embedding with local caching."""
        key = self._cache_key(text)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            embedding = json.loads(cache_file.read_text())
            self._memory_cache[key] = embedding
            return embedding

        # Call API
        embedding = await self._call_api(text, model)

        # Cache to disk and memory
        cache_file.write_text(json.dumps(embedding))
        self._memory_cache[key] = embedding

        return embedding

    async def _call_api(self, text: str, model: str) -> list[float]:
        """Call Vertex AI embedding API."""
        from vertexai.language_models import TextEmbeddingModel
        embed_model = TextEmbeddingModel.from_pretrained(model)
        result = embed_model.get_embeddings([text])
        return result[0].values

    def precompute_bootstrap(self, assets_file: str) -> None:
        """Pre-compute embeddings for all bootstrap assets.

        Run once: python -m src.inventory.precompute_embeddings
        Then local dev never needs to call the embedding API.
        """
        import asyncio
        with open(assets_file) as f:
            assets = json.load(f)
        for asset in assets:
            composite = f"{asset['name']} {asset['description']} {' '.join(asset.get('tags', []))}"
            asyncio.run(self.embed(composite))
        print(f"Cached {len(assets)} embeddings to {self.cache_dir}")
```

---

## 7. Summary of Recommendations

### Phase 0-1 (MVP) — Keep It Simple

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ADK sessions | `InMemorySessionService` | No persistence needed for CLI-based MVP |
| Discussion state | In-memory Python dicts | Fast iteration, no setup |
| Blueprint | In-memory dict | Simple, no persistence |
| Inventory search | `InMemoryInventory` (numpy) | Pre-loaded from JSON, no cloud cost |
| Embeddings | Pre-computed + cached | Run once, cache to disk |
| Local dev | Direct Python execution | No Docker needed yet |

### Phase 2-3 — Add Persistence

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ADK sessions | `InMemorySessionService` (still) | ADK session state is simple enough |
| Discussion state | **Firestore** (our schema) | Async Board needs persistence, UI needs real-time |
| Blueprint | **Firestore** (during discussion) + **GCS** (on finalize) | Transactions + archival |
| Inventory search | `InMemoryInventory` (still) | Scale doesn't demand Vector Search yet |
| Embeddings | Vertex AI API + cache | Live embedding for new assets |
| Local dev | **Firestore Emulator** + fake-gcs-server | Full local stack |

### Phase 4+ — Production Scale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ADK sessions | `DatabaseSessionService` (Firestore) | Production persistence |
| Discussion state | **Firestore** (our schema) | Battle-tested by now |
| Blueprint | **Firestore** + **GCS** | Same as Phase 2-3 |
| Inventory search | **Vertex AI Vector Search** | Production latency and scale |
| Embeddings | Vertex AI API (text-embedding-005) | Production model |
| Local dev | Emulators + in-memory Vector Search fallback | Same interfaces |

### Key Design Principles

1. **Separate ADK session state from domain state.** ADK's session service manages conversation mechanics. Our `DiscussionStore` manages domain state (Blueprint, transcript, decisions). They coexist but don't overlap.

2. **Subcollections for growing data.** The session document stays lean (<10 KB). Transcript, blueprints, patches, and decisions are subcollections that can grow independently.

3. **Transactions for Blueprint patching.** The Dual-Stream State design requires atomic patches with conflict detection. Firestore transactions provide this.

4. **Progressive Vector Search.** Start with in-memory numpy for the MVP. The `InventorySearchBackend` abstraction allows drop-in replacement with Vertex AI Vector Search when scale demands it.

5. **Everything emulatable locally.** Firestore emulator, fake GCS, in-memory search. The only cloud dependency is the LLM API (Vertex AI Gemini), which has no emulator and requires API calls.

6. **State keys as constants.** Follow the `research-agent` pattern of centralizing state key strings to prevent typos across agents.

---

## 8. Firestore Composite Indexes

The following composite indexes are needed for efficient transcript queries:

```
# firestore.indexes.json
{
  "indexes": [
    {
      "collectionGroup": "transcript",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "phase", "order": "ASCENDING"},
        {"fieldPath": "sequence_id", "order": "ASCENDING"}
      ]
    },
    {
      "collectionGroup": "transcript",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "speaker", "order": "ASCENDING"},
        {"fieldPath": "sequence_id", "order": "ASCENDING"}
      ]
    },
    {
      "collectionGroup": "transcript",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "phase", "order": "ASCENDING"},
        {"fieldPath": "message_type", "order": "ASCENDING"},
        {"fieldPath": "sequence_id", "order": "ASCENDING"}
      ]
    },
    {
      "collectionGroup": "transcript",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "speaker", "order": "ASCENDING"},
        {"fieldPath": "phase", "order": "ASCENDING"},
        {"fieldPath": "sequence_id", "order": "ASCENDING"}
      ]
    }
  ]
}
```

These support the `ContextBuilder`'s query patterns: filtering by phase, speaker, message type, and their combinations -- all ordered by sequence_id for chronological reading.
