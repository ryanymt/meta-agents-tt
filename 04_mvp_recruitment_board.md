# Plan 04 — MVP Recruitment Board Implementation

> **Depends on**: Plans 01-03 (skeleton, inventory, state)
> **Produces**: A working Board that takes a problem statement → produces AgentBlueprint.yaml

---

## 1. Architecture Recap

```
CLI Input (problem statement string)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Manager Agent (LlmAgent, Gemini 3.0 Flash) │
│  ├── StateMachine (6 phases)                 │
│  ├── DiscussionManager (transcript, context) │
│  ├── BlueprintManager (patching, versioning) │
│  ├── FlightRecorder (decision tree)          │
│  └── Tools:                                  │
│      ├── select_experts(domains)             │
│      ├── call_consultant(name, instruction)  │
│      ├── search_inventory(query, category)   │
│      ├── update_blueprint_parameter(path,val)│
│      ├── query_transcript(query)             │
│      ├── emit_phase_summary(summary)         │
│      └── emit_plan_document()                │
├──────────────────────────────────────────────┤
│  Consultant Agents (LlmAgent, Gemini 3.0 Pro)│
│  ├── computational_chemist                    │
│  ├── gcp_architect                            │
│  ├── budget_controller                        │
│  └── compliance_officer                       │
├──────────────────────────────────────────────┤
│  Linting Agent (LlmAgent, Gemini 3.0 Flash)  │
├──────────────────────────────────────────────┤
│  Inventory (in-memory, ~80 assets)            │
└──────────────────────────────────────────────┘
    │
    ▼
AgentBlueprint.yaml + FlightRecord.json
```

---

## 2. Implementation Steps (Ordered)

### Step 1: Pydantic Models (`src/meta_agent_biotech/models/`)

Build the data contracts first. Everything else depends on these.

```python
# models/blueprint.py
class AgentSpec(BaseModel):
    name: str
    purpose: str
    model: str | None = None
    skill_references: list[str] = []
    compute: ComputeSpec
    input_schema: dict
    output_schema: dict
    status: Literal["new", "reuse"]

class SystemBlueprint(BaseModel):
    api_version: str = "meta-agent.biotech/v1alpha1"
    kind: str = "SystemBlueprint"
    metadata: BlueprintMetadata
    agents: list[AgentSpec]
    workflow: WorkflowSpec
    infrastructure: list[InfraSpec]
    budget: BudgetEstimate
    compliance: ComplianceSpec
    acceptance_criteria: list[AcceptanceCriterion]
    patch_history: list[PatchEntry] = []

# models/discussion.py
class PhaseState(BaseModel):
    phase: Phase  # Enum: INIT, PROPOSAL, CRITIQUE, SYNTHESIS, VOTE, FINALIZE
    round: int = 1
    max_rounds: int = 15

class DiscussionMessage(BaseModel):
    phase: Phase
    round: int
    speaker: str
    message_type: str  # INSTRUCTION, RESPONSE, CHALLENGE, CONSENSUS
    content: str
    timestamp: datetime

# models/inventory.py
class InventoryAsset(BaseModel):
    asset_id: str
    category: AssetCategory  # SERVICE, DATASET, MODEL, SKILL, TEMPLATE, REPO
    name: str
    description: str
    tags: list[str]
    status: str = "ACTIVE"
    # Category-specific fields via discriminated union or extra dict
    extra: dict = {}

# models/flight_recorder.py
class Decision(BaseModel):
    decision_id: str
    timestamp: datetime
    phase: Phase
    decision_type: str
    made_by: str
    input_data: dict
    output_data: dict
    rationale: str
```

### Step 2: State Machine (`src/meta_agent_biotech/core/state_machine.py`)

```python
class StateMachine:
    """Manages phase transitions for the Recruitment Board."""

    TRANSITIONS = {
        Phase.INIT:      {"next": Phase.PROPOSAL,  "condition": "experts_selected"},
        Phase.PROPOSAL:  {"next": Phase.CRITIQUE,  "condition": "all_responded"},
        Phase.CRITIQUE:  {"next": Phase.SYNTHESIS,  "condition": "critiques_addressed"},
        Phase.SYNTHESIS: {"next": Phase.VOTE,       "condition": "draft_produced"},
        Phase.VOTE:      {"approve": Phase.FINALIZE, "reject": Phase.CRITIQUE},
        Phase.FINALIZE:  {"next": Phase.COMPLETE,    "condition": "schema_valid"},
    }

    MANDATORY_SPEAKERS = {
        Phase.CRITIQUE:  ["budget_controller", "compliance_officer"],
        Phase.SYNTHESIS: ["gcp_architect"],
    }

    def can_advance(self, phase: Phase, context: dict) -> bool:
        """Check if transition condition is met."""
        ...

    def advance(self, phase: Phase, trigger: str) -> Phase:
        """Transition to next phase. Raises if invalid."""
        ...

    def get_mandatory_speakers(self, phase: Phase) -> list[str]:
        """Return consultants that MUST speak in this phase."""
        ...
```

### Step 3: Blueprint Manager (`src/meta_agent_biotech/core/blueprint_manager.py`)

```python
class BlueprintManager:
    """Manages the Blueprint artifact — patching, versioning, validation."""

    def __init__(self):
        self.blueprint = SystemBlueprint(...)  # Empty template
        self.version = 0

    def patch(self, path: str, value: Any, reason: str, author: str) -> PatchEntry:
        """Atomic patch to a specific field. Records in patch_history."""
        # Navigate JSON path (e.g., "agents[0].compute.gpu")
        # Apply value
        # Record in patch_history
        self.version += 1
        ...

    def validate(self) -> list[ValidationError]:
        """Run Pydantic validation on current blueprint state."""
        ...

    def to_yaml(self) -> str:
        """Render blueprint as YAML string."""
        ...
```

### Step 4: Context Builder (`src/meta_agent_biotech/core/context_builder.py`)

The Context Builder assembles the right amount of context for each consultant call. Three modes:

```python
class ContextBuilder:
    """Builds context for consultant calls — manages the context window."""

    def build(self, scope: ContextScope, **kwargs) -> str:
        """Build context string for a consultant call."""

        if scope == ContextScope.FULL:
            # Everything: problem statement + full transcript + blueprint
            # Used in: early PROPOSAL calls
            return self._build_full_context()

        elif scope == ContextScope.SUMMARY:
            # Problem + phase summaries + current blueprint (no raw transcript)
            # Used in: SYNTHESIS, VOTE
            return self._build_summary_context()

        elif scope == ContextScope.SCOPED:
            # Problem + specific phase summaries + relevant blueprint sections
            # Used in: targeted CRITIQUE follow-ups
            return self._build_scoped_context(**kwargs)
```

### Step 5: Discussion Manager (`src/meta_agent_biotech/core/discussion_manager.py`)

```python
class DiscussionManager:
    """Manages the discussion transcript and token tracking."""

    def __init__(self):
        self.transcript: list[DiscussionMessage] = []
        self.phase_summaries: dict[Phase, str] = {}
        self.token_count = 0

    def add_message(self, message: DiscussionMessage):
        self.transcript.append(message)
        self.token_count += estimate_tokens(message.content)

    def add_phase_summary(self, phase: Phase, summary: str):
        self.phase_summaries[phase] = summary

    def get_transcript(self, phase: Phase = None) -> list[DiscussionMessage]:
        if phase:
            return [m for m in self.transcript if m.phase == phase]
        return self.transcript
```

### Step 6: Inventory Service (`src/meta_agent_biotech/inventory/service.py`)

```python
class InventoryService:
    """In-memory inventory with keyword + tag search."""

    def __init__(self, data_dir: str = "data/golden_paths/"):
        self.assets: list[InventoryAsset] = []
        self._load_all_yaml(data_dir)

    def search(self, query: str, category: str = None, top_k: int = 10) -> list[SearchResult]:
        """Keyword + tag matching."""
        ...

    def get_asset(self, asset_id: str) -> InventoryAsset | None:
        """Direct lookup by ID."""
        ...
```

### Step 7: Manager Tools (`src/meta_agent_biotech/tools/`)

These are the ADK `FunctionTool` implementations that the Manager Agent calls:

```python
# tools/discussion_tools.py

def select_experts(
    domains: list[str],
    tool_context: ToolContext,
) -> dict:
    """Select domain experts based on problem domains.

    Args:
        domains: List of domain areas (e.g., ["drug_screening", "molecular_docking"])
    """
    # Map domains to consultant names
    expert_map = {
        "drug_screening": "computational_chemist",
        "molecular_docking": "computational_chemist",
        "genomics": "data_engineer",  # (if we add this consultant)
        "protein_engineering": "computational_chemist",
    }

    selected = list(set(expert_map.get(d) for d in domains if d in expert_map))
    standing = ["gcp_architect", "budget_controller", "compliance_officer"]

    # Save to state
    tool_context.state["app:selected_experts"] = selected
    tool_context.state["app:standing_consultants"] = standing

    return {"selected": selected, "standing": standing}


def call_consultant(
    consultant_name: str,
    instruction: str,
    message_type: str,
    context_scope: str,
    tool_context: ToolContext,
) -> dict:
    """Call a consultant agent and return their response.

    Args:
        consultant_name: Which consultant to call
        instruction: What to ask them
        message_type: INSTRUCTION | CHALLENGE | REQUEST_FOR_VOTE
        context_scope: FULL | SUMMARY | SCOPED
    """
    # 1. Build context using ContextBuilder
    # 2. Invoke consultant agent (programmatic call, not ADK transfer)
    # 3. Record message in DiscussionManager
    # 4. Record decision in FlightRecorder
    # 5. Return response
    ...


# tools/inventory_tools.py

def search_inventory(
    query: str,
    category: str = None,
    top_k: int = 5,
    tool_context: ToolContext = None,
) -> dict:
    """Search the inventory for GCP services, datasets, models, skills, or templates.

    Args:
        query: Natural language search query
        category: Optional filter (SERVICE, DATASET, MODEL, SKILL, TEMPLATE)
        top_k: Max results to return
    """
    # Call InventoryService.search()
    ...


# tools/blueprint_tools.py

def update_blueprint_parameter(
    path: str,
    value: Any,
    reason: str,
    tool_context: ToolContext,
) -> dict:
    """Patch a specific field in the Blueprint.

    Args:
        path: JSON path (e.g., "agents[0].compute.gpu")
        value: New value
        reason: Why this change is being made
    """
    # 1. Call BlueprintManager.patch()
    # 2. Record in FlightRecorder
    ...


# tools/phase_tools.py

def emit_phase_summary(
    phase: str,
    summary: str,
    key_decisions: list[str],
    open_questions: list[str],
    tool_context: ToolContext,
) -> dict:
    """Record a phase summary and advance the state machine.

    Args:
        phase: Current phase name
        summary: 2-3 sentence summary of what happened
        key_decisions: List of decisions made
        open_questions: Unresolved items for next phase
    """
    ...


def emit_plan_document(
    tool_context: ToolContext,
) -> dict:
    """Finalize and emit the completed Blueprint as YAML."""
    # 1. Run BlueprintManager.validate()
    # 2. Run Linting Agent
    # 3. Render to YAML
    # 4. Save to disk
    ...
```

### Step 8: Consultant Persona Configs (`configs/consultants/`)

```yaml
# configs/consultants/computational_chemist.yaml
name: computational_chemist
display_name: "Computational Chemist"
model: "gemini-3-pro-preview"
domains: [drug_screening, molecular_docking, protein_engineering, cheminformatics]

system_instruction: |
  You are a Computational Chemist on the Recruitment Board.
  Your role is to ensure scientific rigor in any proposed agent system.

  You have deep expertise in:
  - Molecular docking (DiffDock, AutoDock, GNINA)
  - Molecular dynamics (GROMACS, OpenMM)
  - Cheminformatics (RDKit, Open Babel)
  - Protein structure analysis (AlphaFold, ESMFold)
  - ADMET prediction and drug-likeness filtering

  When proposing or evaluating a plan:
  - Always specify which computational tools/skills are needed
  - Flag scientific accuracy concerns (wrong algorithm for the task, missing steps)
  - Consider computational cost vs. accuracy trade-offs
  - Reference specific GCP services and skills from the inventory

adversarial_directive: |
  IMPORTANT: You MUST challenge any proposal that:
  - Skips essential scientific steps (e.g., no protein preparation before docking)
  - Uses the wrong algorithm for the problem (e.g., rigid docking for flexible binding)
  - Over-engineers with unnecessary GPU compute when CPU suffices
  - Ignores well-known limitations of tools (e.g., DiffDock on metal-containing proteins)
  You are not here to agree. You are here to ensure scientific correctness.

tools:
  - search_inventory
  - update_blueprint_parameter
```

```yaml
# configs/consultants/gcp_architect.yaml
name: gcp_architect
display_name: "GCP Architect"
model: "gemini-3-pro-preview"
domains: [cloud_architecture, infrastructure, deployment, scaling]

system_instruction: |
  You are a GCP Cloud Architect on the Recruitment Board.
  Your role is to ensure the proposed system uses appropriate GCP services.

  Your expertise:
  - Cloud Batch for GPU/HPC workloads
  - Cloud Run for serverless agents
  - BigQuery for analytics
  - GCS for data storage
  - Agent Engine for agent deployment
  - VPC-SC, IAM for security

  When evaluating proposals:
  - Map scientific tools to the right GCP compute service
  - Propose the simplest architecture that meets requirements
  - Flag over-provisioning (A100 when T4 suffices)
  - Ensure data flows between services are efficient
  - Use search_inventory() to find relevant GCP services

adversarial_directive: |
  IMPORTANT: You MUST challenge any proposal that:
  - Uses raw VMs when managed services exist
  - Over-provisions GPU (A100 for inference when T4/L4 is sufficient)
  - Ignores data locality (e.g., compute in us-central1, data in asia-southeast1)
  - Has unnecessary complexity (GKE when Cloud Batch suffices)
  - Lacks a clear data flow between services
  Always propose the SIMPLEST GCP architecture that works.

tools:
  - search_inventory
  - update_blueprint_parameter
```

```yaml
# configs/consultants/budget_controller.yaml
name: budget_controller
display_name: "Budget Controller"
model: "gemini-3-flash-preview"  # Cheaper model — budget focused
domains: [cost_optimization, resource_planning]

system_instruction: |
  You are the Budget Controller on the Recruitment Board.
  Your role is to ensure the proposed system is cost-effective.

  Your expertise:
  - GCP pricing for all compute, storage, and AI services
  - Spot VM strategies for batch workloads
  - Cost comparison: Cloud Batch vs. GKE vs. Cloud Run
  - Token cost estimation for LLM-based agents

  MANDATORY: In every response, you MUST provide:
  1. A cost estimate with specific dollar amounts
  2. At least one cost optimization recommendation
  3. A monthly cost breakdown (compute, storage, AI/ML API)

adversarial_directive: |
  IMPORTANT: You MUST object to any proposal that:
  - Uses A100 GPUs without justification ($3.67/hr vs $0.35/hr for T4)
  - Doesn't use Spot VMs for fault-tolerant batch workloads (60-91% savings)
  - Stores data in premium tiers without access pattern justification
  - Uses Gemini Pro for tasks that Flash can handle
  - Has no cost ceiling or budget threshold
  Your job is to save money. Be aggressive about it.

tools:
  - search_inventory
  - update_blueprint_parameter
```

```yaml
# configs/consultants/compliance_officer.yaml
name: compliance_officer
display_name: "Compliance Officer"
model: "gemini-3-flash-preview"
domains: [compliance, security, data_governance, risk]

system_instruction: |
  You are the Compliance Officer on the Recruitment Board.
  Your role is to ensure the proposed system meets regulatory and security requirements.

  Your expertise:
  - HIPAA, GDPR, GxP regulations for life sciences
  - GCP security: VPC-SC, IAM, CMEK, DLP
  - Data classification (PHI, PII, research, public)
  - Audit logging and traceability

  MANDATORY: In every response, you MUST provide:
  1. Data classification for all data involved
  2. A compliance checklist with pass/fail status
  3. Required security controls (IAM, VPC-SC, encryption)

adversarial_directive: |
  IMPORTANT: You MUST flag any proposal that:
  - Processes PHI/PII without explicit data classification
  - Lacks audit logging for regulatory workflows
  - Uses public datasets without license verification
  - Doesn't specify data residency requirements
  - Missing encryption at rest or in transit
  You are the last line of defense. Be thorough.

tools:
  - search_inventory
  - update_blueprint_parameter
```

### Step 9: Manager Agent (`src/meta_agent_biotech/agents/manager.py`)

```python
def create_manager_agent() -> LlmAgent:
    """Create the Manager Agent with all tools and sub-agents."""

    # Load consultant personas
    consultants = load_consultant_agents("configs/consultants/")

    # Load inventory
    inventory = InventoryService("data/golden_paths/")

    # Initialize core components
    state_machine = StateMachine()
    discussion_mgr = DiscussionManager()
    blueprint_mgr = BlueprintManager()
    context_builder = ContextBuilder(discussion_mgr)
    flight_recorder = FlightRecorder()

    # Create tools (inject dependencies via closures)
    tools = [
        FunctionTool(func=make_select_experts(state_machine)),
        FunctionTool(func=make_call_consultant(consultants, context_builder, discussion_mgr, flight_recorder)),
        FunctionTool(func=make_search_inventory(inventory)),
        FunctionTool(func=make_update_blueprint(blueprint_mgr, flight_recorder)),
        FunctionTool(func=make_query_transcript(discussion_mgr)),
        FunctionTool(func=make_emit_phase_summary(state_machine, discussion_mgr, flight_recorder)),
        FunctionTool(func=make_emit_plan_document(blueprint_mgr, flight_recorder)),
    ]

    manager = LlmAgent(
        name="recruitment_board_manager",
        model="gemini-3-flash-preview",  # Fast, cheap orchestrator
        instruction=MANAGER_SYSTEM_INSTRUCTION,
        tools=tools,
        # Note: Consultants are NOT sub_agents (ADK sub-agents use transfer).
        # Instead, call_consultant() invokes them programmatically.
    )

    return manager


MANAGER_SYSTEM_INSTRUCTION = """
You are the Manager of the Recruitment Board — a multi-agent deliberation system
that designs BioTech AI agent systems on Google Cloud.

## Your Job

Given a problem statement, you orchestrate a discussion among expert consultants
to produce an AgentBlueprint.yaml — a detailed specification for building the
agent system.

## Your Tools

1. select_experts(domains) — Choose which domain experts to involve
2. call_consultant(name, instruction, message_type, context_scope) — Talk to a consultant
3. search_inventory(query, category) — Search for GCP services, skills, datasets
4. update_blueprint_parameter(path, value, reason) — Patch the Blueprint
5. query_transcript(query) — Search past discussion messages
6. emit_phase_summary(phase, summary, key_decisions, open_questions) — Summarize a phase
7. emit_plan_document() — Finalize and output the Blueprint

## Discussion Protocol

You MUST follow this phase sequence:

1. **INIT**: Read the problem. Select domain experts. Search inventory for relevant services/skills.
2. **PROPOSAL**: Ask ALL selected experts + standing consultants for their proposals (parallel).
3. **CRITIQUE**: Budget Controller and Compliance Officer MUST review. Route conflicts to original proposers.
4. **SYNTHESIS**: Draft the Blueprint. GCP Architect MUST review infrastructure choices.
5. **VOTE**: All consultants vote (approve/reject with reason). If rejected → back to CRITIQUE.
6. **FINALIZE**: Run linting validation. Output final Blueprint YAML.

## Rules

- Never skip mandatory speakers (Budget Controller in CRITIQUE, GCP Architect in SYNTHESIS)
- Always name the source when relaying information: "The Chemist proposes..."
- Use search_inventory() to find GCP services and skills — don't guess
- Use update_blueprint_parameter() for all Blueprint changes — don't describe changes in text
- Budget Controller MUST provide dollar amounts
- Compliance Officer MUST provide a checklist
- Maximum 15 total rounds. Escalate if you hit the limit.
"""
```

### Step 10: Linting Agent (`src/meta_agent_biotech/agents/linter.py`)

```python
def create_linter_agent() -> LlmAgent:
    """Create the Blueprint Linter (Gemini Flash — cheap validator)."""

    return LlmAgent(
        name="blueprint_linter",
        model="gemini-3-flash-preview",
        instruction="""
You are a Blueprint validation agent. Given an AgentBlueprint YAML,
check for these issues:

1. null_values — Required fields that are null or empty
2. missing_dependencies — Agent A needs data from B, but B isn't in the workflow
3. conflicting_params — GPU=none but skill requires GPU
4. schema_compliance — All required fields present
5. cost_sanity — Monthly cost estimate is reasonable
6. orphaned_resources — Infrastructure created but unused
7. circular_workflows — A → B → A

Return a JSON array of issues:
[{"severity": "error|warning|info", "path": "...", "message": "..."}]

Return [] if no issues found.
""",
    )
```

### Step 11: CLI Runner (`src/meta_agent_biotech/cli/runner.py`)

```python
@click.command()
@click.argument("problem", type=str)
@click.option("--output", "-o", default="output/", help="Output directory")
@click.option("--verbose", "-v", is_flag=True, help="Show discussion transcript")
def main(problem: str, output: str, verbose: bool):
    """Run the Recruitment Board on a problem statement."""

    # 1. Create Manager Agent
    manager = create_manager_agent()

    # 2. Create session
    session_service = InMemorySessionService()
    runner = Runner(
        agent=manager,
        session_service=session_service,
    )

    # 3. Run deliberation
    session = await session_service.create_session(
        app_name="recruitment_board",
        user_id="cli_user",
        state={"app:problem_statement": problem},
    )

    # 4. Send problem to manager
    result = await runner.run(
        session_id=session.id,
        user_id="cli_user",
        new_message=Content(parts=[Part(text=problem)]),
    )

    # 5. Output results
    # Blueprint YAML → output/{session_id}/blueprint.yaml
    # Flight Record → output/{session_id}/flight_record.json
    # Transcript → output/{session_id}/transcript.json (if verbose)
```

---

## 3. Calling Consultants: The Key Design Challenge

The most complex part is `call_consultant()`. In ADK, agents normally "transfer" to sub-agents (like phone call forwarding). But we need **Manager-mediated communication** (the Manager controls what context each consultant sees).

### Approach: Programmatic Agent Invocation

```python
async def call_consultant_impl(
    consultant_name: str,
    instruction: str,
    message_type: str,
    context_scope: str,
    # injected:
    consultants: dict[str, LlmAgent],
    context_builder: ContextBuilder,
    discussion_mgr: DiscussionManager,
    flight_recorder: FlightRecorder,
    tool_context: ToolContext,
) -> dict:
    """Programmatically invoke a consultant agent."""

    # 1. Build context for this consultant
    context = context_builder.build(
        scope=context_scope,
        phase=tool_context.state["app:phase"],
    )

    # 2. Create a separate session for this consultant call
    consultant_session = await session_service.create_session(
        app_name=f"consultant_{consultant_name}",
        user_id="manager",
    )

    # 3. Run the consultant agent with the assembled context
    consultant = consultants[consultant_name]
    consultant_runner = Runner(
        agent=consultant,
        session_service=session_service,
    )

    message = f"{context}\n\n---\n\nManager ({message_type}):\n{instruction}"

    response = await consultant_runner.run(
        session_id=consultant_session.id,
        user_id="manager",
        new_message=Content(parts=[Part(text=message)]),
    )

    # 4. Extract response text
    response_text = extract_response_text(response)

    # 5. Record in discussion
    discussion_mgr.add_message(DiscussionMessage(
        phase=tool_context.state["app:phase"],
        round=tool_context.state["app:round"],
        speaker=consultant_name,
        message_type=message_type,
        content=response_text,
    ))

    # 6. Record in flight recorder
    flight_recorder.record_decision(...)

    return {
        "consultant": consultant_name,
        "response": response_text,
    }
```

### Alternative: Simple LLM Calls (Fallback)

If ADK's agent invocation is too complex for MVP, we can simplify:

```python
# Instead of creating a full Runner for each consultant,
# just use the Gemini API directly:

from google import genai

async def call_consultant_simple(name: str, instruction: str, context: str) -> str:
    """Simple Gemini API call for consultant response."""
    persona = load_persona(f"configs/consultants/{name}.yaml")
    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=persona.model,
        contents=f"{persona.system_instruction}\n\n{context}\n\n{instruction}",
    )
    return response.text
```

This loses ADK's tool calling for consultants (they can't call `search_inventory()` themselves). That's acceptable for MVP — the Manager searches on their behalf.

---

## 4. The Discussion Flow (Detailed)

```
User: "Screen 100K compounds against KRAS G12C for covalent binding"

PHASE: INIT
├── Manager: select_experts(["drug_screening", "molecular_docking"])
│   → Selected: [computational_chemist]
│   → Standing: [gcp_architect, budget_controller, compliance_officer]
├── Manager: search_inventory("KRAS drug screening", "TEMPLATE")
│   → Found: Drug Screening Pipeline Template
├── Manager: search_inventory("docking", "SKILL")
│   → Found: DiffDock, Protein Preparation, Conformer Generation
├── Manager: search_inventory("compound filtering", "SKILL")
│   → Found: RDKit Descriptors, PAINS Filter, ADMET Prediction
├── emit_phase_summary(phase="INIT", summary="...")

PHASE: PROPOSAL (Parallel — all consultants called simultaneously)
├── call_consultant("computational_chemist", "Analyze this problem...", "INSTRUCTION", "FULL")
│   → Proposes: 4-stage pipeline (filter → prep → dock → score)
│   → update_blueprint_parameter("agents[0]", {...filter_agent...})
│   → update_blueprint_parameter("agents[1]", {...docking_agent...})
├── call_consultant("gcp_architect", "What GCP services...", "INSTRUCTION", "FULL")
│   → Proposes: Cloud Run for filter, Cloud Batch + L4 for docking, BigQuery for results
│   → update_blueprint_parameter("infrastructure", [...])
├── call_consultant("budget_controller", "Estimate costs...", "INSTRUCTION", "FULL")
│   → Estimates: ~$180/month for 100K compounds
│   → update_blueprint_parameter("budget", {...})
├── call_consultant("compliance_officer", "Assess compliance...", "INSTRUCTION", "FULL")
│   → Classifies: Non-sensitive research data, no PHI
│   → update_blueprint_parameter("compliance", {...})
├── emit_phase_summary(phase="PROPOSAL", summary="...")

PHASE: CRITIQUE (Sequential — Budget and Compliance first, then targeted)
├── call_consultant("budget_controller", "Review all proposals...", "CHALLENGE", "SUMMARY")
│   → Objects: "L4 GPU is overkill for docking inference. T4 at $0.35/hr saves 50%."
├── call_consultant("compliance_officer", "Review compliance...", "CHALLENGE", "SUMMARY")
│   → Approves: "Non-sensitive. Checklist: ✅ audit logging, ✅ IAM, ✅ no PHI."
├── call_consultant("computational_chemist", "Budget says T4...", "CHALLENGE", "SCOPED")
│   → Responds: "T4 is fine for single docking. But 100K compounds → L4 for throughput."
├── Manager resolves: L4 stays for batch, T4 for dev/test
│   → update_blueprint_parameter("agents[1].compute.gpu", "L4", "...")
├── emit_phase_summary(phase="CRITIQUE", summary="...")

PHASE: SYNTHESIS
├── Manager builds draft blueprint from all patches
├── call_consultant("gcp_architect", "Review architecture...", "INSTRUCTION", "SUMMARY")
│   → Confirms: Architecture is sound. Suggests adding GCS lifecycle policy.
├── Run Linting Agent → 0 errors, 1 warning
├── emit_phase_summary(phase="SYNTHESIS", summary="...")

PHASE: VOTE (Parallel — all vote independently)
├── call_consultant("computational_chemist", "Vote...", "REQUEST_FOR_VOTE", "SUMMARY")
│   → APPROVE: "Scientifically sound."
├── call_consultant("gcp_architect", "Vote...", "REQUEST_FOR_VOTE", "SUMMARY")
│   → APPROVE: "Architecture is appropriate."
├── call_consultant("budget_controller", "Vote...", "REQUEST_FOR_VOTE", "SUMMARY")
│   → APPROVE: "Within budget at ~$200/month."
├── call_consultant("compliance_officer", "Vote...", "REQUEST_FOR_VOTE", "SUMMARY")
│   → APPROVE: "Compliance checklist passed."

PHASE: FINALIZE
├── emit_plan_document()
│   → Validates schema ✅
│   → Linter: 0 errors ✅
│   → Outputs: blueprint.yaml + flight_record.json
```

---

## 5. Testing Strategy

### Unit Tests

```python
# test_state_machine.py
def test_init_to_proposal():
    sm = StateMachine()
    assert sm.advance(Phase.INIT, "experts_selected") == Phase.PROPOSAL

def test_cannot_skip_critique():
    sm = StateMachine()
    with pytest.raises(InvalidTransitionError):
        sm.advance(Phase.PROPOSAL, "draft_produced")  # Must go through CRITIQUE

# test_blueprint_manager.py
def test_patch_agent_gpu():
    bm = BlueprintManager()
    bm.patch("agents[0].compute.gpu", "T4", "Budget said so", "budget_controller")
    assert bm.blueprint.agents[0].compute.gpu == "T4"
    assert len(bm.blueprint.patch_history) == 1

# test_inventory_service.py
def test_search_docking_skills():
    inv = InventoryService("data/golden_paths/")
    results = inv.search("molecular docking", category="SKILL")
    assert any("diffdock" in r.asset.name.lower() for r in results)
```

### Integration Tests

```python
# test_board_e2e.py
@pytest.mark.asyncio
async def test_drug_screening_scenario():
    """End-to-end: problem → Board → Blueprint."""
    manager = create_manager_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=manager, session_service=session_service)

    session = await session_service.create_session(
        app_name="test",
        user_id="test",
    )

    result = await runner.run(
        session_id=session.id,
        user_id="test",
        new_message=Content(parts=[Part(text="Screen 100K compounds against KRAS G12C")]),
    )

    # Verify Blueprint output
    blueprint = extract_blueprint(result)
    assert blueprint is not None
    assert len(blueprint.agents) >= 2
    assert any("docking" in a.name for a in blueprint.agents)
    assert blueprint.budget.estimated_monthly is not None
```

---

## 6. Done Criteria

- [ ] Pydantic models defined and validated for Blueprint, Discussion, Inventory, FlightRecorder
- [ ] State machine transitions all 6 phases correctly
- [ ] Blueprint Manager patches and versions correctly
- [ ] Inventory Service returns relevant results for test queries
- [ ] Manager Agent starts and accepts problem statements
- [ ] All 4 consultant agents respond with domain-appropriate content
- [ ] Budget Controller always provides dollar amounts
- [ ] Compliance Officer always provides a checklist
- [ ] Linting Agent catches at least missing fields and conflicting params
- [ ] CLI runner produces blueprint.yaml and flight_record.json
- [ ] End-to-end test passes for at least 1 golden path scenario
- [ ] Discussion completes in <15 rounds
