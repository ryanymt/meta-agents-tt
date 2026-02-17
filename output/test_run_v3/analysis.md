# Test Run v2 vs v3 — Comparative Analysis

> **Date**: 2026-02-17
> **v2**: Claude cross-check failed (404 — wrong region)
> **v3**: Claude cross-check working (`claude-sonnet-4-5@20250929` via Vertex AI us-east5)

---

## High-Level Metrics

| Metric | v2 | v3 | Delta |
|--------|-----|-----|-------|
| Agents | 5 | **6** | +1 (ScreeningSpecialist added) |
| Workflow steps | 6 | 6 | Same |
| Budget | $200.40/mo | **$125/mo** | **-37.5%** |
| Blueprint patches | 23 | **37** | +61% more refinement |
| Flight recorder decisions | 42 | **57** | +36% more deliberation |
| Discussion messages | 13 | 13 | Same |
| Vote | 3-1 (budget rejected) | **4-0 unanimous** | Cleaner consensus |
| Cross-check | Failed (404) | **Working** (13,240 chars) | Key improvement |
| Budget breakdown | All zeros | **Itemized** ($110 compute, $12 storage, $2 AI, $1 network) | Now useful |
| Model specified | None (missing) | **gemini-3-flash-preview** on all agents | Fixed |

---

## Claude Cross-Check Impact (The Big Difference)

**v2**: Claude cross-check hit a 404 error (wrong region). It logged `[Cross-check error: ...]` and the pipeline continued without independent review. Response was 429 chars (the error message).

**v3**: Claude (`claude-sonnet-4-5@20250929` on us-east5) produced a **13,240-character adversarial review** that REJECTED the blueprint. The Manager then made **11 patches in SYNTHESIS** to address Claude's findings:

| Claude Finding | Manager's Fix |
|---------------|--------------|
| Agents had no model specified or used deprecated `gemini-1.5-flash` | Upgraded all 6 agents to `gemini-3-flash-preview` |
| Network egress `deny-all` blocks ChEMBL/PDB downloads | Enabled `allow-internet` for data fetchers |
| MD timeout too short (1800s for GROMACS) | Increased to 7200s (2hr safety margin) |
| Budget breakdown was all zeros | Itemized: compute $110, storage $12, AI $2, networking $1 |
| ScreeningSpecialist description too vague | Added scientific detail: "Vina docking with distance constraint (<4A to Cys12 SG)" |

---

## Architecture Differences

### v2 Pipeline (5 agents, linear)

```
data_prep → smina_docking → result_processor → gnina → result_processor → gromacs
            (16 CPU, 32Gi)                     (L4 GPU)                   (L4 GPU)
```

### v3 Pipeline (6 agents, two-stage docking)

```
CheminformaticsSpecialist → StructuralBiologist → ScreeningSpecialist → DockingSpecialist → DynamicsEngineer → ResultsArchivist
(1 CPU, 2Gi)               (1 CPU, 2Gi)          (4 CPU, 8Gi, Spot)    (T4 GPU, Spot)      (T4 GPU, 32Gi)     (1 CPU, 2Gi)
                                                   Vina: 50K → 2K        GNINA: 2K → top      1ns MD on top 10
```

### Key Architectural Changes

- **v3 split receptor prep into its own agent** (StructuralBiologist) instead of bundling it with data prep
- **v3 adopted Two-Stage Docking** (Vina fast screen → GNINA rescore on top 2K) — v2 ran GNINA on all filtered compounds
- **v3 downsized GPU from L4 to T4** for both docking and MD — v2 used L4 GPUs
- **v3 downsized lightweight agents** to 1 CPU / 2Gi — v2 had 4 CPU / 16Gi even for data prep
- **v3 reduced MD to 1ns** (stability check) vs v2's 10ns (full simulation) — scientifically justified as "pre-reactive complex stability" check

---

## Scientific Quality

| Aspect | v2 | v3 |
|--------|-----|-----|
| Docking strategy | Single-stage (Smina → GNINA) | **Two-stage** (Vina fast → GNINA precise on top 2K) |
| Covalent binding | Geometric filter (Cys12 < 4.0A) | Distance constraint (<4A to Cys12 SG) — **more specific** |
| MD scope | 10ns on top 20 poses | **1ns on top 10** — pragmatic for validation |
| MD parameterization | GAFF2/AM1-BCC via Acpype (added late in VOTE) | Pre-reactive stability focus (designed from CRITIQUE) |
| Receptor prep | Retain GDP/Mg cofactors | Same, plus **separate StructuralBiologist agent** |
| Container images | Pinned SHA256 digests, Artifact Registry mirrors | Not specified (gap) |
| Checkpointing | GROMACS checkpoint to GCS for Spot VM safety | Not explicit (gap) |

---

## Compliance & Security

| Control | v2 | v3 |
|---------|-----|-----|
| Data classification | `non-sensitive` | `research-internal` (stricter) |
| IAM Least Privilege | Yes | Yes |
| VPC-SC | Yes | Yes |
| CMEK | Yes (with specific key path) | Yes |
| Audit Logging | Yes (DATA_ACCESS specified) | Yes |
| Container digest pinning | Yes | No (gap) |
| Artifact Registry mirrors | Yes (5 agents pointed to AR) | No (gap) |
| Network egress | `deny-all` everywhere | `allow-internet` for data fetchers (correct) |

---

## Voting

**v2**: Budget Controller voted `false` — recorded as a rejection but the pipeline still finalized. The Chemist added Acpype parameterization as a late conditional fix during VOTE phase.

**v3**: All 4 voted `true` — **clean unanimous approval**. The Manager addressed budget concerns proactively during CRITIQUE/SYNTHESIS before reaching VOTE.

---

## What v3 Does Better (Thanks to Claude Cross-Check)

1. **Budget**: 37.5% cheaper through Two-Stage Docking and right-sized compute
2. **Model governance**: All agents have explicit, current model versions
3. **Network policy**: Data-fetching agents correctly allow internet access
4. **Cost transparency**: Itemized breakdown instead of zeros
5. **Clean consensus**: 4-0 vote vs 3-1
6. **More deliberation**: 57 decisions / 37 patches show deeper refinement

## What v2 Did Better (Without Cross-Check)

1. **Container governance**: SHA256 digest pinning + Artifact Registry mirrors — v3 missed this entirely
2. **Compliance specificity**: CMEK key path, DATA_ACCESS audit level, container digest pinning
3. **Spot VM safety**: Explicit GROMACS checkpoint-resume wrapper logic
4. **Infra resources**: v2 specified Artifact Registry as an infrastructure resource

---

## Root Cause Analysis

The Claude cross-check pushed v3 to fix model versions, network policies, and budget — but the GCP Architect in v3 was focused on addressing Claude's findings and didn't independently add the infra hardening that the v2 Architect contributed during its SYNTHESIS phase.

The non-deterministic nature of LLM deliberation means each run discovers different issues. Ideally we'd want **both** v2's infra hardening and v3's cross-check fixes.

## Recommended Improvements

1. **Add infrastructure checklist to GCP Architect system prompt**: Artifact Registry, digest pinning, checkpoint logic should be mandatory considerations, not left to chance
2. **Add a post-cross-check validation pass**: After addressing Claude's findings, run a final compliance/infra check to catch items that may have been deprioritized
3. **Persist cross-check findings as structured data**: Parse Claude's CRITICAL/HIGH/MEDIUM/LOW findings into the flight recorder for trend analysis across runs
4. **Consider running cross-check earlier**: Moving it to end of CRITIQUE (before SYNTHESIS) would give more time to address findings
