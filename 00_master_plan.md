# Master Implementation Plan

> **Goal**: Build a working MVP — from `plans/` to a running Recruitment Board that produces GCP-native AgentBlueprints for BioTech problems.
>
> **Approach**: Bottom-up. Foundation first, then the Board, then validate with real scenarios.

---

## Build Order (Critical Path)

```
Step 1: Project Skeleton + ADK Scaffold
   ↓
Step 2: Schemas (Pydantic models + JSON Schema)
   ↓
Step 3: Inventory Data (GCP golden paths + skills metadata)
   ↓
Step 4: Inventory Service (in-memory search_inventory())
   ↓
Step 5: Consultant Personas (YAML configs + adversarial directives)
   ↓
Step 6: State Machine + Discussion Manager
   ↓
Step 7: Manager Agent + Tools (ADK LlmAgent)
   ↓
Step 8: Consultant Agents (4 ADK sub-agents)
   ↓
Step 9: Blueprint Linter + Flight Recorder
   ↓
Step 10: CLI Runner + End-to-End Testing
   ↓
Step 11: Validate with 3 Golden Path Scenarios
```

---

## What to Build vs. What to Defer

### Build Now (MVP)

| Component | Why Now |
|-----------|---------|
| Pydantic models for Blueprint, Discussion, Inventory | Everything depends on these contracts |
| In-memory inventory with GCP golden paths | Board needs `search_inventory()` to propose GCP-native solutions |
| Skill metadata (YAML manifests, not full SKILL.md indexing) | Board needs skill names + descriptions for recommendations |
| Manager Agent + 4 consultants (ADK) | This IS the product |
| State machine (6 phases) | Controls Board deliberation flow |
| Blueprint Linter (Gemini Flash) | Validates output quality |
| Flight Recorder (JSON logs) | Traceability for debugging |
| CLI runner | Local testing without UI |

### Defer (Post-MVP)

| Component | Why Later |
|-----------|-----------|
| Vector Search / Vertex AI embeddings | In-memory keyword + tag search is enough for ~200 assets |
| Firestore persistence | In-memory state works for single sessions |
| Web UI / Problem Framing Agent | CLI is sufficient for MVP validation |
| Development Team (code gen) | Blueprint output is the MVP deliverable, not deployed agents |
| Full SKILL.md knowledge indexing | Board only needs skill names/descriptions, not full docs |
| Agent Engine deployment | Local `adk run` is fine for MVP |
| GCS / Artifact Registry | No file artifacts in MVP |

---

## Key Decisions (Pre-Validated)

### 1. Should We Build More Skills?

**No, not yet.** The `claude-scientific-skills` repo already has 131 skills. For MVP:
- Index skill **metadata** (name, description, tags, runtime requirements) — not full SKILL.md content
- The Board references skills by name when proposing Blueprints
- Full SKILL.md reading is only needed by the Code Gen Agent (Phase 3)

**Action**: Create a `skills_catalog.yaml` with curated metadata for the ~30 most relevant skills. Don't index all 131 — just the ones that map to our golden path scenarios.

### 2. Should We Map Out What GCP Offers?

**Yes, but scoped.** Don't catalog everything Google Cloud has. Catalog what matters for BioTech:

- **Compute**: Cloud Batch (GPU jobs), Cloud Run (serverless), GKE (orchestration)
- **Storage**: GCS (files), BigQuery (analytics), Filestore (NFS for HPC)
- **AI/ML**: Gemini models, MedGemma, TxGemma, AlphaFold API, Model Garden
- **Data**: BigQuery public datasets (1000 Genomes, ClinVar, gnomAD, FDA FAERS)
- **Healthcare**: Healthcare API (FHIR/DICOM), VPC-SC, DLP API
- **DevOps**: Artifact Registry, Cloud Build, Secret Manager

**Action**: Create `data/golden_paths/` YAML files with ~60–80 curated GCP assets. Each asset has: name, description, tags, cost_hint, use_cases, constraints.

### 3. How Do We Store Data/State?

**MVP: All in-memory. Production: Firestore + GCS.**

| Layer | MVP Storage | Production Storage | What's Stored |
|-------|-------------|-------------------|---------------|
| Inventory | Python dict in memory | Firestore + Vector Search | Asset catalog (~200 entries) |
| Discussion state | ADK Session state (in-memory) | ADK VertexAiSessionService | Phase, round, transcript, blueprint draft |
| Blueprint artifact | Python dict → YAML file output | Firestore document + GCS | The final plan document |
| Flight Recorder | JSON file on disk | Firestore collection | Decision tree, metrics |
| Consultant context | ADK Session events | ADK VertexAiSessionService | Per-consultant conversation history |

See `plans/03_data_state_storage.md` for detailed state architecture.

### 4. Which ADK Patterns to Use?

Based on ADK v1.24.1 source inspection:

| ADK Concept | How We Use It |
|-------------|---------------|
| `LlmAgent` | Manager Agent + each Consultant Agent |
| `InMemorySessionService` | MVP session persistence |
| `InMemoryMemoryService` | MVP memory (cross-session, not needed for MVP) |
| `Session.state` with `app:` prefix | Discussion state (phase, round, blueprint) |
| `Session.state` with `temp:` prefix | Per-turn scratch data (discarded between turns) |
| `FunctionTool` | All 7 Manager tools + consultant tools |
| `Runner` | Orchestrates agent execution with services |
| `before_agent_callback` | State machine phase enforcement |
| `after_agent_callback` | Auto-lint after phase transitions |

**NOT using** (MVP):
- `SequentialAgent` / `ParallelAgent` — We hand-code the orchestration for more control
- `VertexAiSessionService` — MVP uses in-memory
- `VertexAiMemoryBankService` — No cross-session memory needed
- Agent-to-agent transfer — Manager mediates all communication via tools

---

## Timeline Estimate (Not a Commitment)

```
Week 1:  Steps 1-3 (skeleton, schemas, inventory data)
Week 2:  Steps 4-6 (inventory service, personas, state machine)
Week 3:  Steps 7-8 (Manager + consultants, ADK integration)
Week 4:  Steps 9-11 (linter, recorder, CLI, golden path testing)
```

---

## Success Criteria (MVP)

1. **Input**: A problem statement string (e.g., "Screen 100K compounds against KRAS G12C")
2. **Output**: A valid `AgentBlueprint.yaml` that:
   - Proposes GCP-native services (BigQuery, Cloud Batch — not raw VMs)
   - References real skills from the catalog
   - Has cost estimates from the Budget Controller
   - Has a compliance checklist from the Compliance Officer
   - Passes the Blueprint Linter with zero errors
3. **Process**: Discussion completes in <15 rounds with full Flight Recorder trace
4. **Validation**: Works for 3 golden path scenarios (drug screening, variant calling, protein engineering)

---

## File Index

| Plan | Description |
|------|-------------|
| [01_project_setup.md](01_project_setup.md) | ADK scaffold, pyproject.toml, directory structure |
| [02_inventory_and_knowledge.md](02_inventory_and_knowledge.md) | GCP catalog, skills metadata, golden path data |
| [03_data_state_storage.md](03_data_state_storage.md) | State management for each layer (MVP → production) |
| [04_mvp_recruitment_board.md](04_mvp_recruitment_board.md) | Step-by-step Board implementation |
| [05_golden_path_scenarios.md](05_golden_path_scenarios.md) | 3 target validation scenarios |
| [06_mvp_dev_team.md](06_mvp_dev_team.md) | Simplified Dev Team plan (post-MVP) |
