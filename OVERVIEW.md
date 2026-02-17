# Meta-Agent BioTech — System Overview

> **A meta-agentic framework that designs and deploys BioTech AI agents on Google Cloud.**
> Users describe biotech problems in natural language; the system assembles expert consultants, architects a cloud-native solution, and builds the agents automatically.

---

## The Problem

Building AI-powered biotech pipelines on GCP today requires coordinating across multiple specialties — computational chemistry, genomics, cloud architecture, compliance, cost optimization — then hand-coding agents, infrastructure, and deployment scripts. This takes weeks of expert time per pipeline.

## The Vision

**Describe a problem. Get deployed agents.**

```
"Screen 50,000 compounds against KRAS G12C for covalent inhibitors"
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Meta-Agent BioTech                                           │
│                                                               │
│  1. Problem Framing    → Clarify requirements                 │
│  2. Recruitment Board  → Expert panel designs the solution    │
│  3. Development Team   → Agents build and deploy agents       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  5 deployed agents on Vertex AI Agent Engine
  (data prep → docking → scoring → validation → results)
  Budget: $200/month  |  Compliance: checked  |  Reusable via A2A
```

Each solved problem adds reusable agents to an **Inventory**, creating a flywheel — future problems are faster and cheaper because existing agents can be composed rather than rebuilt.

---

## Architecture: Five Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0: Problem Framing Agent                         │
│  Conversational agent. Clarifies requirements.          │
│  User selects domain experts.                           │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Recruitment Board  ← MVP (working today)     │
│  Multi-agent deliberation. 6 phases.                    │
│  Produces AgentBlueprint.yaml (the "plan").             │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Development Team                              │
│  5-agent CI/CD pipeline. Reads Blueprint.               │
│  Generates code, infra (Terraform), tests, deploys.     │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Agent Engine (Vertex AI)                      │
│  Production hosting. A2A protocol for inter-agent       │
│  communication. Auto-registered in Inventory.           │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Gemini Enterprise                             │
│  End-user scientists access agents via natural language. │
│  They never see the underlying infrastructure.          │
└─────────────────────────────────────────────────────────┘
```

**Human checkpoints**: Users approve the expert panel (L0→L1) and the final plan (L1→L2). Everything else is automated.

---

## Layer 1: The Recruitment Board (MVP)

This is the core of the system and what we've built first. Given a biotech problem, the Board assembles a panel of AI consultants who debate, critique, and converge on a cloud-native architecture.

### How It Works

**The Board is a structured deliberation, not a free-form chat.** A Manager Agent orchestrates the discussion using a state machine with 6 phases:

```
INIT → PROPOSAL → CRITIQUE → SYNTHESIS → VOTE → FINALIZE
```

| Phase | What Happens | Key Mechanic |
|-------|-------------|--------------|
| **1. INIT** | Manager analyzes the problem, selects domain experts | Always includes: GCP Architect, Data Engineer, Budget Controller, Compliance Officer |
| **2. PROPOSAL** | Each expert independently proposes agents and infrastructure | **Parallel** — prevents anchoring bias (experts don't see each other's ideas) |
| **3. CRITIQUE** | Budget Controller costs it. Compliance flags risks. Experts cross-critique. | **Adversarial by design** — Budget must find waste, Compliance must find risk |
| **4. SYNTHESIS** | Manager drafts unified Blueprint. Experts confirm their areas. Claude (independent model) cross-checks. | **Dual-model review** — Gemini designs, Claude audits |
| **5. VOTE** | Every member votes approve/reject with rationale | Unanimous required. Objections loop back to CRITIQUE |
| **6. FINALIZE** | Emit structured AgentBlueprint.yaml | Machine-readable handover to Development Team |

### The Consultant Roster

**Domain Specialists** (selected per problem):

| Role | Example Expertise |
|------|------------------|
| Computational Chemist | Molecular docking, QSAR, compound filtering |
| Genomics Specialist | Variant calling, NGS, single-cell analysis |
| Structural Biologist | AlphaFold, protein design, cryo-EM |
| Pharmacologist | ADMET, PK/PD, drug safety |
| Bioprocess Engineer | Fermentation, lab automation, GMP |
| Clinical Data Scientist | Clinical trials, real-world evidence |
| Systems Biologist | Pathway analysis, multi-omics |

**Standing Members** (always present):

| Role | Adversarial Mandate |
|------|-------------------|
| GCP Solutions Architect | "Flag single points of failure. Insist on health checks." |
| Data Engineer | "Unidirectional data flow. Every dataset needs a schema." |
| Budget Controller | "NEVER approve without a cheaper alternative. Always calculate cost-per-unit." |
| Compliance & Security | "ASSUME data is sensitive until proven otherwise. Never guess on classification." |

### Key Design Decisions

**Manager-mediated communication**: Consultants never talk directly to each other. The Manager quotes and attributes ("Budget Controller objects to A100 usage at $3.67/hr. Computational Chemist, can T4 handle this?"). This prevents side-channels and ensures the Manager can't "forget" critical input.

**Dual-Stream State**: Two parallel data streams prevent information loss:
- **Narrative stream** (lossy): Phase summaries for LLM context — compact but may drop details
- **Blueprint artifact** (lossless): Structured YAML patched atomically via tools — never summarized, only updated

**Independent cross-check**: During SYNTHESIS, Claude (via Vertex AI Model Garden) reviews the Blueprint produced by Gemini-based consultants. Different model family = different blind spots caught.

**Flight Recorder**: Every decision is logged with rationale, inputs, and outputs. Not for replay (LLMs are non-deterministic), but for traceability — "why was T4 chosen over A100?" is always answerable.

### What the Board Produces

An `AgentBlueprint.yaml` — a structured, machine-readable plan:

```yaml
api_version: meta-agent.biotech/v1alpha1
kind: SystemBlueprint
metadata:
  name: kras-g12c-screening-pipeline
  version: 1.0.0
  description: Virtual screening pipeline for KRAS G12C inhibitors

agents:
  - name: data_prep_agent          # Fetches compounds, filters, preps receptor
    runtime: { type: cloud_batch, cpu: 4, memory: 16Gi, spot_vm: true }
  - name: smina_docking_agent      # Docks compounds against target (GPU)
    runtime: { type: cloud_batch, gpu: nvidia-tesla-t4, spot_vm: true }
  - name: gnina_agent              # ML-based rescoring of top hits
    runtime: { type: cloud_batch, gpu: nvidia-tesla-t4 }
  - name: gromacs_agent            # MD validation of top candidates
    runtime: { type: cloud_batch, gpu: nvidia-l4, memory: 32Gi }
  - name: result_processor_agent   # Aggregates results to BigQuery
    runtime: { type: cloud_run, cpu: 2, memory: 8Gi }

workflow:
  steps: [data_prep → smina_docking → gnina → gromacs → result_processor]

budget: { estimated_monthly: $200 }
compliance: { data_classification: non-sensitive, controls: [audit_logging] }

patch_history:  # Full deliberation trail (23 patches in test run)
  - { author: budget_controller, change: "GPU A100 → T4", reason: "75% cost savings" }
  - { author: compliance_officer, change: "Added VPC-SC", reason: "Data sovereignty" }
```

---

## Layer 2: The Development Team (Designed, Not Yet Built)

Once a user approves the Blueprint, a 5-agent CI/CD pipeline builds and deploys everything:

```
Blueprint.yaml
     │
     ├─── Code Gen Agent ──────── Writes Python ADK agent code
     │    (reads SKILL.md knowledge docs + reference scripts + existing GCP repos)
     │
     ├─── Infra Gen Agent ─────── Writes Terraform (GCS, BigQuery, IAM, networking)
     │
     ├─── Runtime Architect ───── Selects Docker base images, configures GPU/CUDA
     │
     ├─── Test Agent ──────────── Compile check → unit tests → smoke tests
     │    (3-strike rule: fix errors up to 3x, then escalate to human)
     │
     └─── Dev Manager ─────────── Orchestrates the above, triggers Cloud Build
          (deploy to Agent Engine, register in Inventory)
```

### Skill-Guided Code Generation

The Code Gen Agent doesn't import pre-built functions. It reads **knowledge documents** (SKILL.md files from a library of 131+ scientific skills) plus reference implementations from existing GCP repos, then writes fresh code. Like a developer reading documentation before building.

| Knowledge Source | What It Provides |
|-----------------|-----------------|
| SKILL.md (131 skills) | Scientific context, capabilities, limitations, best practices |
| Reference scripts | Code patterns showing library usage (RDKit, DiffDock, etc.) |
| User's GCP repos | Working Cloud Batch configs, BigQuery schemas, GPU setup |

---

## The Inventory: Flywheel for Reuse

Every deployed agent is registered in an **Inventory** — a semantic catalog searchable by the Recruitment Board.

```
First problem:  "Screen compounds against KRAS G12C"
  → Board designs 5 agents from scratch
  → All 5 registered in Inventory

Second problem: "Screen compounds against EGFR T790M"
  → Board finds 3 of 5 agents are reusable (data prep, docking, results)
  → Only 2 need modification
  → Cost: 40% of first run
```

The Inventory pre-ships with a **Day 1 Content Pack**: 23 GCP services, 10 AI/ML models, 15+ BigQuery public datasets, and 131 scientific skill knowledge docs — so the Board proposes cloud-native solutions from the very first session.

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Framework | Google ADK v1.24.1 | Native GCP integration, Agent Engine deployment |
| Primary Model | Gemini (2.5 Flash for MVP) | Cost-effective, strong tool use |
| Cross-Check Model | Claude Sonnet 4.5 (via Vertex AI) | Independent adversarial review |
| State Store | Firestore | Session state, inventory, discussion logs |
| Hosting | Cloud Run (meta-system), Agent Engine (deployed agents) | Serverless, scales to zero |
| Batch Compute | Cloud Batch | GPU workloads (docking, MD simulations) |
| Infrastructure as Code | Terraform | Reproducible GCP resource provisioning |
| Inter-Agent Protocol | A2A (Agent-to-Agent) | Standardized discovery and reuse |

---

## Current Status: MVP

### What's Built and Working

The **Recruitment Board (Layer 1) MVP** is functional end-to-end:

- **28 Python source files**, ~2,700 lines of code
- **55 unit tests**, all passing
- **10 Manager tools**: select_experts, call_consultant, update_blueprint, emit_phase_summary, emit_plan_document, search_inventory, query_discussion, escalate_to_human, record_vote, cross_check_blueprint
- **State machine** enforcing 6-phase deliberation with mandatory speakers
- **Claude cross-check** via Vertex AI Model Garden (us-east5)
- **Flight Recorder** capturing full decision traces
- **CLI runner** for local testing

### Test Run Results (KRAS G12C Screening)

```
Input:   "Design a virtual screening pipeline for identifying KRAS G12C inhibitors"
Output:  AgentBlueprint.yaml with:
         - 5 agents (data prep, docking, rescoring, MD validation, results)
         - 6 workflow steps with dependencies
         - Budget: $200/month (Spot VMs, T4 GPUs)
         - 23 blueprint patches (full deliberation trail)
         - 43 flight recorder decisions
         - 13 discussion messages across 6 phases
```

### Roadmap

| Phase | Status | What |
|-------|--------|------|
| **0. Foundation** | ~85% | Schemas, design docs, plans, GCP catalog |
| **1. Recruitment Board MVP** | Working | Multi-agent deliberation → Blueprint |
| **2. Problem Framing + UI** | Not started | Web interface, conversational agent |
| **3. Development Team** | Not started | Auto-build agents from Blueprint |
| **4. Full Inventory** | Not started | Vector Search, reuse-first logic |
| **5. Agent Engine** | Not started | Production deployment |
| **6. Polish & Scale** | Not started | Monitoring, advanced workflows |

---

## What We're Looking For Feedback On

1. **Architecture**: Does the 5-layer separation make sense? Are we missing a layer?
2. **Deliberation Protocol**: Is adversarial-by-design the right approach? Too rigid? Too loose?
3. **Dual-Model Review**: Is Claude cross-checking Gemini's output valuable, or overkill?
4. **Skill-Guided Code Gen**: Reading knowledge docs and writing fresh code vs. other approaches?
5. **Reuse Strategy**: Is the Inventory flywheel compelling? What about versioning concerns?
6. **Scope**: Is the full L0-L4 pipeline too ambitious? Should we stop at Blueprint generation?
