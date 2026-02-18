# GCP Service Catalog for Life Sciences / BioTech

> **Status**: Reference Catalog
> **Last Updated**: 2026-02-15
> **Purpose**: Comprehensive catalog of Google Cloud Platform services relevant to BioTech customers. Used to bootstrap the Inventory system's Day 1 Content Pack.
> **Knowledge Cutoff**: Verified through May 2025 GCP documentation + known GA releases.

---

## 1. Compute Services

### 1.1 Cloud Batch

| Field | Value |
|-------|-------|
| **Service** | Cloud Batch |
| **API** | `batch.googleapis.com` |
| **Terraform** | `google_cloud_batch_job` |
| **Console** | https://console.cloud.google.com/batch |
| **Pricing** | No additional charge beyond underlying Compute Engine resources |
| **Status** | GA |

**Description**: Fully managed batch processing service for scheduling, queueing, and executing large-scale compute workloads. No cluster management required. Natively supports GPU workloads, Spot VMs, and container-based jobs.

**Biotech Use Cases**:
- Molecular docking (DiffDock, Gnina, AutoDock Vina) -- GPU batch
- Variant calling (DeepVariant, GATK) -- CPU batch
- Molecular dynamics (GROMACS, OpenMM, AMBER) -- GPU batch
- Compound library filtering (RDKit) -- CPU batch
- AlphaFold structure prediction -- GPU batch
- Nextflow/Cromwell workflow execution

**Key Features**:
- Task arrays for embarrassingly parallel workloads (up to 10,000 tasks per job)
- GPU support: NVIDIA T4, L4, A100 (40GB/80GB), H100
- Spot VM support (60-91% discount for fault-tolerant jobs)
- Container-based jobs with GCS FUSE mounting
- Automatic retry with exponential backoff
- Job priority and scheduling
- Lifecycle hooks (pre/post task scripts)

**GPU Machine Types for Biotech**:

| Machine Type | GPU | vCPUs | Memory | Use Case | On-Demand $/hr |
|-------------|-----|-------|--------|----------|---------------|
| `n1-standard-4` + T4 | NVIDIA T4 (16GB) | 4 | 15 GB | Light docking, inference | ~$0.35 + $0.19 |
| `g2-standard-8` | NVIDIA L4 (24GB) | 8 | 32 GB | DiffDock, medium inference | ~$0.84 |
| `g2-standard-24` | NVIDIA L4 x2 | 24 | 96 GB | Batch docking, training | ~$2.52 |
| `a2-highgpu-1g` | NVIDIA A100 40GB | 12 | 85 GB | AlphaFold, large MD sims | ~$3.67 |
| `a2-ultragpu-1g` | NVIDIA A100 80GB | 12 | 170 GB | AlphaFold multimer, training | ~$5.07 |
| `a3-highgpu-1g` | NVIDIA H100 80GB | 26 | 170 GB | Large-scale training | ~$8.68 |

**Spot VM Discounts**: 60-91% off on-demand pricing. Recommended for all fault-tolerant bio workloads (docking, MD, screening).

**Quick Start**:
```bash
gcloud batch jobs submit my-docking-job \
  --location=us-central1 \
  --config=job.json
```

---

### 1.2 Cloud Run

| Field | Value |
|-------|-------|
| **Service** | Cloud Run |
| **API** | `run.googleapis.com` |
| **Terraform** | `google_cloud_run_v2_service` / `google_cloud_run_v2_job` |
| **Pricing** | Pay per request + CPU/memory/time |
| **Status** | GA |

**Description**: Fully managed serverless container platform. Supports both services (always-on or scale-to-zero HTTP endpoints) and jobs (one-off or scheduled tasks).

**Biotech Use Cases**:
- Agent API endpoints (lightweight inference, data retrieval)
- Webhook receivers for pipeline orchestration
- REST API wrappers for scientific tools
- Small-batch compound filtering
- Agent Engine backend services

**Key Features**:
- Scale to zero (cost-effective for intermittent workloads)
- Up to 8 vCPUs, 32 GiB memory per instance (2nd gen)
- GPU support (NVIDIA L4) in Cloud Run -- GA since late 2024
- Cloud Run Jobs for batch-like workloads without managing Batch
- VPC connector for private network access
- Custom domains and IAM-based authentication
- Automatic HTTPS, load balancing, and autoscaling

**Pricing Estimates**:
- CPU: $0.00002400 / vCPU-second
- Memory: $0.00000250 / GiB-second
- Requests: $0.40 / million requests
- GPU (L4): ~$0.000289 / GPU-second

---

### 1.3 Google Kubernetes Engine (GKE)

| Field | Value |
|-------|-------|
| **Service** | Google Kubernetes Engine |
| **API** | `container.googleapis.com` |
| **Terraform** | `google_container_cluster`, `google_container_node_pool` |
| **Pricing** | Free (Autopilot management) or $0.10/hr (Standard cluster fee) + node costs |
| **Status** | GA |

**Description**: Managed Kubernetes for container orchestration. Two modes: Autopilot (fully managed, pay-per-pod) and Standard (manage node pools).

**Biotech Use Cases**:
- Long-running scientific services (Nextflow Tower, JupyterHub)
- Multi-tenant research platforms
- GPU-accelerated ML training clusters
- Kubeflow Pipelines for ML workflows
- Argo Workflows for complex DAG pipelines

**Key Features**:
- Autopilot mode: fully managed pods with GPU support
- GPU node pools (T4, L4, A100, H100)
- Spot/Preemptible nodes for cost savings
- GKE Sandbox for workload isolation
- Workload Identity for secure GCP service access
- Node auto-provisioning

---

### 1.4 Compute Engine

| Field | Value |
|-------|-------|
| **Service** | Compute Engine |
| **API** | `compute.googleapis.com` |
| **Terraform** | `google_compute_instance` |
| **Pricing** | Per-second billing, Spot/CUD/SUD discounts available |
| **Status** | GA |

**Description**: IaaS virtual machines. Foundation for Cloud Batch and GKE. Direct VM use for interactive research workloads.

**Biotech-Relevant Machine Families**:

| Family | Best For | Example |
|--------|----------|---------|
| N2/N2D | General bioinformatics (BLAST, BWA) | `n2-standard-32` (32 vCPUs, 128GB) |
| C2/C2D | CPU-intensive (MD, assembly) | `c2-standard-60` (60 vCPUs, 240GB) |
| C3 | Latest gen, best per-core performance | `c3-standard-44` |
| M2/M3 | Memory-intensive (genome assembly, de novo) | `m3-megamem-128` (128 vCPUs, 1.9TB) |
| G2 | GPU (NVIDIA L4) | `g2-standard-8` |
| A2 | GPU (NVIDIA A100) | `a2-highgpu-1g` |
| A3 | GPU (NVIDIA H100) | `a3-highgpu-8g` |

**Discount Options**:
- Spot VMs: 60-91% discount (can be preempted)
- Committed Use Discounts (CUD): 1-yr (37%) or 3-yr (55%) commitment
- Sustained Use Discounts (SUD): Automatic 20-30% off for sustained usage

---

## 2. AI/ML Services

### 2.1 Vertex AI Platform

| Field | Value |
|-------|-------|
| **Service** | Vertex AI |
| **API** | `aiplatform.googleapis.com` |
| **Console** | https://console.cloud.google.com/vertex-ai |
| **Status** | GA |

**Description**: Unified AI/ML platform encompassing model training, deployment, tuning, evaluation, and generative AI services.

**Key Components for Biotech**:

| Component | Purpose | API |
|-----------|---------|-----|
| Model Garden | Pre-trained model catalog (150+) | `aiplatform.googleapis.com` |
| Generative AI | Gemini model API | `generativelanguage.googleapis.com` |
| Vertex AI Agent Engine | Managed agent hosting | `aiplatform.googleapis.com` |
| Custom Training | Train custom models | `aiplatform.googleapis.com` |
| Prediction (Endpoints) | Deploy models for inference | `aiplatform.googleapis.com` |
| Vector Search | ANN-based similarity search | `aiplatform.googleapis.com` |
| Feature Store | Managed feature engineering | `aiplatform.googleapis.com` |
| Pipelines | Kubeflow-based ML pipelines | `aiplatform.googleapis.com` |
| Experiments | Track ML experiments | `aiplatform.googleapis.com` |
| Model Evaluation | AutoSxS, safety eval | `aiplatform.googleapis.com` |

---

### 2.2 Vertex AI Model Garden -- Life Science Models

#### Gemini Foundation Models

| Model | Model ID | Context Window | Best For | Pricing |
|-------|----------|---------------|----------|---------|
| Gemini 3.0 Flash | `gemini-3-flash-preview` | 1M tokens | Agent orchestration, fast tasks | TBD (preview pricing) |
| Gemini 3.0 Pro | `gemini-3-pro-preview` | 1M tokens | Complex reasoning, analysis | TBD (preview pricing) |
| Gemini 1.5 Pro | `gemini-1.5-pro` | 2M tokens | Long-context analysis (fallback) | $1.25/1M input, $5.00/1M output |
| Gemini 1.5 Flash | `gemini-1.5-flash` | 1M tokens | High-volume processing (fallback) | $0.075/1M input, $0.30/1M output |

> **Note**: We use Gemini 3.0 Preview models (`gemini-3-pro-preview`, `gemini-3-flash-preview`) as the primary models. Gemini 1.5 models are listed as fallbacks.

#### Life Science Specialized Models

| Model | Type | Access | Description | Hardware | Status |
|-------|------|--------|-------------|----------|--------|
| **MedGemma** | Medical LLM | Model Garden (click-through license) | Gemini-based model fine-tuned for medical text understanding. Supports clinical QA, biomedical NER, drug interaction analysis, clinical note summarization. | L4 or A100 GPU | GA (2025) |
| **TxGemma** | Therapeutics | Model Garden (click-through license) | Specialized for therapeutic development: toxicity prediction, ADMET property estimation, drug-target interaction, molecular property prediction. | L4 or A100 GPU | Preview/GA (2025) |
| **Med-PaLM 2** | Medical QA | Restricted access (Google Cloud partnership) | Medical question answering achieving expert-level performance on USMLE-style questions. Research-focused. | Managed API | Limited Access |
| **AlphaFold** | Protein Structure | Model Garden (open) | DeepMind's protein structure prediction model. Predicts 3D protein structures from amino acid sequences. AlphaFold DB contains 200M+ predicted structures. | A100 GPU (multimer), T4/L4 (monomer) | GA (open-source) |
| **DeepVariant** | Variant Calling | Open source / Model Garden | Deep learning-based variant caller for genomic data. Achieves state-of-the-art accuracy for SNPs and indels from NGS data (Illumina, PacBio, ONT). | CPU or GPU | GA (open-source) |
| **ESM-2 / ESMFold** | Protein Language Model | Hugging Face / Model Garden | Meta's protein language model for sequence embeddings, structure prediction, and function annotation. | GPU recommended | Open source |

**MedGemma Details**:
```
Model Garden ID: medgemma
Variants: medgemma-4b (4B params), medgemma-27b (27B params)
Capabilities:
  - Medical question answering (multi-choice, open-ended)
  - Clinical note summarization
  - Biomedical named entity recognition (genes, proteins, diseases, drugs)
  - Drug interaction analysis
  - Radiology report generation (multimodal variants)
Deploy command:
  gcloud ai endpoints deploy-model ENDPOINT_ID \
    --model=medgemma-4b \
    --region=us-central1 \
    --machine-type=g2-standard-12 \
    --accelerator-type=NVIDIA_L4
```

**TxGemma Details**:
```
Model Garden ID: txgemma
Variants: txgemma-2b, txgemma-9b, txgemma-27b
Capabilities:
  - Molecular property prediction (LogP, solubility, toxicity)
  - ADMET property estimation
  - Drug-target binding affinity prediction
  - Compound classification (BACE, HIV, Tox21, ClinTox)
  - Natural language interface to molecular data
Unique: Can process both SMILES strings and natural language in the same prompt
```

**AlphaFold on GCP**:
```
Container: us-docker.pkg.dev/deepmind-environments/alphafold/alphafold:latest
Hardware: A100 GPU (40GB minimum for monomers, 80GB for multimers)
Databases: 2.2TB genetic databases (BFD, MGnify, PDB70, UniRef90)
Deployment: Cloud Batch job with GCS for input/output
Estimated cost: $5-50 per prediction (depending on protein size)
AlphaFold DB: https://alphafold.ebi.ac.uk (200M+ pre-computed structures)
BigQuery public dataset: bigquery-public-data.deepmind_alphafold
```

**DeepVariant on GCP**:
```
Container: google/deepvariant:latest
Hardware: CPU (n1-standard-16+) or GPU (T4 for acceleration)
Input: BAM/CRAM aligned reads + reference genome
Output: VCF/gVCF variant calls
Deployment: Cloud Batch or Nextflow on Cloud Batch
Estimated cost: $0.50-5.00 per whole genome (30x coverage, using Spot VMs)
Companion tools: GLnexus (joint genotyping), hap.py (benchmarking)
```

---

### 2.3 Vertex AI Agent Engine

| Field | Value |
|-------|-------|
| **Service** | Vertex AI Agent Engine (formerly Agent Builder / Reasoning Engine) |
| **API** | `aiplatform.googleapis.com` |
| **Terraform** | `google_vertex_ai_agent` (emerging) |
| **Status** | GA (core), some features in Preview |

**Description**: Managed runtime for deploying AI agents built with Google ADK (Agent Development Kit), LangChain, LangGraph, or CrewAI. Handles scaling, session management, and infrastructure.

**Capabilities**:
- Deploy agents as managed API endpoints
- Session and state management (built-in)
- Tool execution runtime
- Multi-agent orchestration
- Grounding with Google Search, Vertex AI Search, or custom data stores
- Streaming responses
- Observability (Cloud Trace integration)

**Agent Development Kit (ADK)**:
```
Framework: google-adk (Python SDK)
Agent Types:
  - LlmAgent: Single LLM-powered agent with tools
  - SequentialAgent: Ordered pipeline of sub-agents
  - ParallelAgent: Concurrent execution of sub-agents
  - LoopAgent: Iterative refinement with exit conditions
  - CustomAgent: Fully custom control flow
Tool Types:
  - FunctionTool: Python function as tool
  - AgentTool: Another agent as a tool (sub-agent)
  - GoogleSearchTool: Grounding with Google Search
  - VertexAiSearchTool: Grounding with enterprise data
Session:
  - InMemorySessionStore (development)
  - VertexAiSessionStore (production, managed)
  - FirestoreSessionStore (custom persistence)
```

**Deployment**:
```python
# Deploy agent to Agent Engine
from google.adk.deploy import deploy_to_agent_engine

agent_engine = deploy_to_agent_engine(
    agent=my_agent,
    project="my-project",
    location="us-central1",
    display_name="my-biotech-agent",
    requirements=["google-adk", "rdkit-pypi", "pandas"],
)
# Returns: projects/my-project/locations/us-central1/reasoningEngines/{id}
```

**A2A Protocol Support**:
- Agent-to-Agent (A2A) is an open protocol for inter-agent communication
- Agent Engine supports A2A discovery via `/.well-known/agent.json` Agent Cards
- Agents can discover and call other agents via A2A task lifecycle
- A2A task states: submitted, working, input-required, completed, failed, canceled

**Quotas (Approximate)**:
| Resource | Default Limit | Notes |
|----------|--------------|-------|
| Agents per project | 100 | Can request increase |
| Sessions per agent | 10,000 concurrent | Configurable |
| Tool execution timeout | 600 seconds | Per tool call |
| Max agent response time | 120 seconds (streaming) | Configurable |
| Request payload size | 10 MB | Input/output |

**Pricing**:
- Agent Engine itself: No separate charge (charged for underlying Vertex AI API calls)
- Model inference: Standard Gemini API pricing per token
- Grounding: Charged per grounding query
- Storage (sessions): Included

---

### 2.4 Vertex AI Vector Search

| Field | Value |
|-------|-------|
| **Service** | Vertex AI Vector Search (formerly Matching Engine) |
| **API** | `aiplatform.googleapis.com` |
| **Terraform** | `google_vertex_ai_index`, `google_vertex_ai_index_endpoint` |
| **Pricing** | Shard-hour based (~$0.12/shard-hour) |
| **Status** | GA |

**Description**: Managed approximate nearest neighbor (ANN) search service. Handles billions of vectors with sub-millisecond latency.

**Biotech Use Cases**:
- Semantic search over the Inventory catalog
- Molecular similarity search (fingerprint vectors)
- Literature search (embedding-based)
- Patient similarity matching

**Embedding Model**: `text-embedding-005` (768 dimensions, latest and most capable)

---

## 3. Data Services

### 3.1 BigQuery

| Field | Value |
|-------|-------|
| **Service** | BigQuery |
| **API** | `bigquery.googleapis.com` |
| **Terraform** | `google_bigquery_dataset`, `google_bigquery_table` |
| **Pricing** | Storage: $0.02/GB/mo (active), $0.01/GB/mo (long-term). Queries: $6.25/TB scanned (on-demand) or flat-rate slots |
| **Status** | GA |

**Description**: Serverless, highly scalable data warehouse with built-in ML (BigQuery ML), geospatial analytics, and a rich ecosystem of public datasets.

**Biotech Use Cases**:
- Genomic variant analytics at scale
- Clinical trial data analysis
- Drug adverse event surveillance (FAERS)
- Docking results warehousing
- Population-scale genomics (1000 Genomes, gnomAD)
- Patent landscape analysis
- OMICS data integration

**Key Features for Biotech**:
- BigQuery ML: Train ML models in SQL (logistic regression, XGBoost, deep learning, LLM inference)
- `ML.GENERATE_EMBEDDING()`: Generate embeddings directly in SQL
- `ML.GENERATE_TEXT()`: Call Gemini from SQL queries
- BI Engine: In-memory analysis acceleration
- Materialized views for precomputed aggregations
- Row/Column-level security for PHI/PII
- Data masking and data policies
- Public datasets (free storage, pay for queries)
- Scheduled queries
- BigQuery DataFrames (pandas-like API at warehouse scale)

---

### 3.2 BigQuery Public Datasets -- Life Sciences

All datasets below are in the `bigquery-public-data` project and storage is **free**. You pay only for query processing ($6.25/TB scanned on-demand).

#### Genomics & Genetics

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **1000 Genomes (Phase 3)** | `bigquery-public-data.human_genome_variants` | `1000_genomes_phase_3` | ~500 GB | Comprehensive catalog of human genetic variation. 84.4M variants from 2,504 individuals across 26 populations. |
| **gnomAD** | `bigquery-public-data.gnomAD` | `gnomad_genomes_v3`, `gnomad_exomes_v2_1_1` | ~5 TB | Genome Aggregation Database. Allele frequencies from 76,156+ genomes and 125,748+ exomes. Gold standard for variant frequency. |
| **ClinVar** | `bigquery-public-data.human_genome_variants` | `clinvar_hg38`, `clinvar_hg19` | ~2 GB | NCBI's database of clinically relevant genomic variants. Pathogenicity classifications, clinical significance. |
| **NCBI Gene Info** | `bigquery-public-data.ncbi` | `gene_info` | ~1 GB | Gene nomenclature, descriptions, and cross-references for all organisms. |
| **Human Genome Reference** | `bigquery-public-data.human_genome_variants` | `hg19_reference`, `hg38_reference` | ~3 GB | GRCh37/GRCh38 reference genome sequences. |

**Sample Query -- 1000 Genomes**:
```sql
SELECT
  reference_name,
  start_position,
  reference_bases,
  alternate_bases[OFFSET(0)].alt AS alt,
  call[OFFSET(0)].genotype
FROM `bigquery-public-data.human_genome_variants.1000_genomes_phase_3`
WHERE reference_name = 'chr17'
  AND start_position BETWEEN 43044295 AND 43125483  -- BRCA1 region
LIMIT 100
```

**Sample Query -- gnomAD Allele Frequency**:
```sql
SELECT
  chromosome, pos, ref, alt,
  af AS allele_frequency,
  ac AS allele_count,
  an AS allele_number
FROM `bigquery-public-data.gnomAD.gnomad_genomes_v3`
WHERE chromosome = 'chr17'
  AND pos BETWEEN 43044295 AND 43125483
ORDER BY af DESC
LIMIT 50
```

#### Drug Safety & Regulatory

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **FDA FAERS** | `bigquery-public-data.fda_food` and `bigquery-public-data.fda_drug` | `food_events`, `drug_label` | ~10 GB | FDA Adverse Event Reporting System. Food and drug adverse events, drug labels. |
| **FDA Drug Enforcement** | `bigquery-public-data.fda_drug` | `drug_enforcement` | ~500 MB | Drug recall and enforcement actions. |
| **Open FDA** | `bigquery-public-data.fda_food` | `food_events` | ~2 GB | Structured food adverse event reports. |

**Sample Query -- FDA Drug Adverse Events**:
```sql
SELECT
  safetyreportid,
  patient.drug[OFFSET(0)].medicinalproduct AS drug_name,
  patient.reaction[OFFSET(0)].reactionmeddrapt AS adverse_event,
  receivedate
FROM `bigquery-public-data.fda_drug.drug_event`
WHERE UPPER(patient.drug[OFFSET(0)].medicinalproduct) LIKE '%SOTORASIB%'
ORDER BY receivedate DESC
LIMIT 100
```

#### Patents & Intellectual Property

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **Google Patents** | `patents-public-data.patents` | `publications`, `publications_latest` | ~1 TB | Full-text patent publications from 100+ patent offices worldwide. Claims, abstracts, classifications (CPC/IPC). |
| **Google Patents Research** | `patents-public-data.patents` | `research_results` | ~100 GB | Enriched patent data with extracted entities, similarity scores. |

**Sample Query -- Patent Search**:
```sql
SELECT
  publication_number,
  title_localized[OFFSET(0)].text AS title,
  abstract_localized[OFFSET(0)].text AS abstract,
  filing_date,
  grant_date,
  cpc[OFFSET(0)].code AS primary_cpc
FROM `patents-public-data.patents.publications`
WHERE
  REGEXP_CONTAINS(
    abstract_localized[OFFSET(0)].text,
    r'(?i)KRAS.*G12C.*inhibitor'
  )
  AND country_code = 'US'
ORDER BY filing_date DESC
LIMIT 50
```

#### Public Health & Epidemiology

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **COVID-19 Open Data** | `bigquery-public-data.covid19_open_data` | `covid19_open_data` | ~5 GB | Global COVID-19 epidemiology data with demographics, hospitalization, vaccination. |
| **WHO Global Health** | `bigquery-public-data.world_bank_health_population` | `health_nutrition_population` | ~1 GB | World Bank health indicators across countries and years. |

#### Protein Structure

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **AlphaFold DB** | `bigquery-public-data.deepmind_alphafold` | `structures` (metadata) | ~50 GB | Metadata for 200M+ AlphaFold-predicted protein structures. Includes pLDDT confidence scores, UniProt mappings. |

**Sample Query -- AlphaFold Structures**:
```sql
SELECT
  uniprot_accession,
  organism_scientific_name,
  global_metric_value AS avg_plddt,
  sequence_length
FROM `bigquery-public-data.deepmind_alphafold.structures`
WHERE organism_scientific_name = 'Homo sapiens'
  AND global_metric_value > 80  -- High confidence
ORDER BY sequence_length DESC
LIMIT 100
```

#### Other Life Science-Adjacent Datasets

| Dataset | Project.Dataset | Key Tables | Approx Size | Description |
|---------|----------------|------------|-------------|-------------|
| **PubMed (via NLM)** | `bigquery-public-data.nlm_rxnorm` | Various tables | ~5 GB | RxNorm drug terminology. For PubMed abstracts, use the NLM API or third-party BQ mirrors. |
| **Open Targets Genetics** | Available via Open Targets Platform | Various | ~100 GB+ | Genetic association data (requires separate access, not directly in BQ public data). |
| **ENCODE** | Not directly in BQ public data | N/A | N/A | Functional genomics data; accessible via GCS public buckets. |
| **TCGA (Cancer Genome Atlas)** | `isb-cgc-bq.TCGA` (ISB-CGC project) | Various per cancer type | ~50 TB | Cancer genomics. Requires ISB-CGC access, not directly in bigquery-public-data. |

#### GCS Public Buckets (Life Sciences)

These are not BigQuery datasets but publicly accessible Cloud Storage buckets:

| Bucket | Description | Access |
|--------|-------------|--------|
| `gs://genomics-public-data` | Reference genomes, 1000 Genomes BAMs | Public |
| `gs://gcp-public-data--gnomad` | gnomAD VCF files | Public |
| `gs://gcp-public-data--broad-references` | Broad Institute reference data (dbSNP, Mills, etc.) | Public |
| `gs://deepmind-alphafold` | AlphaFold predicted structures (PDB/mmCIF) | Public |
| `gs://gatk-best-practices` | GATK reference bundles | Public |

---

### 3.3 Healthcare API

| Field | Value |
|-------|-------|
| **Service** | Cloud Healthcare API |
| **API** | `healthcare.googleapis.com` |
| **Terraform** | `google_healthcare_dataset`, `google_healthcare_fhir_store`, `google_healthcare_dicom_store`, `google_healthcare_hl7_v2_store` |
| **Pricing** | FHIR: $0.03/10K ops (structured), DICOM: storage + ops, HL7v2: $0.05/10K messages |
| **Status** | GA |
| **Compliance** | HIPAA BAA, HITRUST, SOC 1/2/3, ISO 27001, FedRAMP |

**Description**: Managed API for storing, querying, and managing healthcare data in FHIR, DICOM, and HL7v2 formats. Provides de-identification, consent management, and data harmonization.

**Supported Standards**:

| Standard | Version | Use Case |
|----------|---------|----------|
| **FHIR** | R4 (STU), R4B | Electronic health records, patient data, clinical observations, genomic data (Genomics Reporting IG) |
| **DICOM** | Standard | Medical imaging (CT, MRI, X-ray, pathology). DICOMweb REST API. |
| **HL7v2** | v2.x (all versions) | Legacy system integration, lab results, ADT messages |

**Key Features**:
- **FHIR Store**: RESTful FHIR R4 server with search, batch, and transaction support
- **DICOM Store**: DICOMweb (WADO-RS, STOW-RS, QIDO-RS) for medical imaging
- **HL7v2 Store**: Parse, store, and route HL7v2 messages
- **De-identification**: Built-in PHI removal for FHIR and DICOM data
- **Consent Management**: FHIR Consent resources with enforcement
- **FHIR Search**: Full FHIR search parameter support with custom search parameters
- **Pub/Sub Notifications**: Trigger downstream processing on FHIR resource changes
- **BigQuery Streaming**: Auto-export FHIR resources to BigQuery for analytics
- **Data Harmonization**: Transform data between standards

**Genomics via FHIR**:
The Healthcare API supports FHIR Genomics Reporting Implementation Guide:
- `MolecularSequence` resource for raw sequence data
- `Observation` (genomics) for variant observations
- `DiagnosticReport` (genomics) for clinical genomic reports
- Integration with ClinVar and other annotation sources

**API Endpoint Pattern**:
```
https://healthcare.googleapis.com/v1/
  projects/{project}/locations/{location}/datasets/{dataset}/
  fhirStores/{fhir_store}/fhir/{resourceType}/{id}
```

**Quick Start**:
```bash
# Create a FHIR store
gcloud healthcare fhir-stores create my-fhir-store \
  --dataset=my-dataset \
  --location=us-central1 \
  --version=R4

# Create a DICOM store
gcloud healthcare dicom-stores create my-dicom-store \
  --dataset=my-dataset \
  --location=us-central1
```

---

### 3.4 Cloud Storage (GCS)

| Field | Value |
|-------|-------|
| **Service** | Cloud Storage |
| **API** | `storage.googleapis.com` |
| **Terraform** | `google_storage_bucket` |
| **Pricing** | Standard: $0.020/GB/mo, Nearline: $0.010, Coldline: $0.004, Archive: $0.0012 |
| **Status** | GA |

**Description**: Object storage with global availability, strong consistency, and integration with all GCP services.

**Biotech Use Cases**:
- PDB/mmCIF protein structure files
- FASTQ/BAM/CRAM sequencing data
- SDF/MOL2 compound libraries
- Model weights and checkpoints
- Pipeline intermediate files
- Results archival

**Key Features for Biotech**:
- Cloud Storage FUSE: Mount buckets as local filesystems (critical for Cloud Batch)
- Autoclass: Automatic storage class transitions for cost optimization
- Object Lifecycle Management: Auto-delete intermediate files
- Customer-Managed Encryption Keys (CMEK) for sensitive data
- Retention policies for regulatory compliance (GxP)
- Requester-pays buckets for shared datasets
- Dual-region and multi-region options for resilience

**Storage Class Strategy for Biotech**:
| Data Type | Storage Class | Rationale |
|-----------|--------------|-----------|
| Active pipeline data | Standard | Frequent access during processing |
| Reference genomes | Standard or Nearline | Periodic access |
| Raw sequencing data | Nearline or Coldline | Access after initial processing |
| Regulatory archives | Archive + Retention Lock | Compliance requirement, rare access |

---

### 3.5 Firestore

| Field | Value |
|-------|-------|
| **Service** | Cloud Firestore |
| **API** | `firestore.googleapis.com` |
| **Terraform** | `google_firestore_database` |
| **Pricing** | Reads: $0.06/100K, Writes: $0.18/100K, Storage: $0.15/GB/mo |
| **Status** | GA |

**Description**: Serverless NoSQL document database with real-time sync, offline support, and strong consistency.

**Biotech Use Cases (per our architecture)**:
- Inventory metadata storage (full asset details, schemas, configs)
- Agent session state
- Discussion logs and transcripts
- Blueprint document storage
- User preferences and configuration

**Key Features**:
- Document model (JSON-like, nested)
- Real-time listeners (streaming updates)
- Automatic multi-region replication
- TTL (time-to-live) for expiring documents
- Composite indexes for complex queries
- Transaction support (serializable isolation)
- Integration with Firebase Auth for user-facing apps

---

## 4. Workflow & Orchestration Services

### 4.1 Cloud Workflows

| Field | Value |
|-------|-------|
| **Service** | Cloud Workflows |
| **API** | `workflows.googleapis.com` |
| **Terraform** | `google_workflows_workflow` |
| **Pricing** | $0.01 per 1,000 internal steps; $0.025 per 1,000 external HTTP calls |
| **Status** | GA |

**Description**: Serverless workflow orchestration for multi-step processes. Executes steps sequentially or in parallel with error handling, retries, and conditional logic.

**Biotech Use Cases**:
- Multi-step agent pipelines (submit Cloud Batch job -> wait -> process results -> notify)
- Async orchestration of long-running scientific jobs
- SYNC / ASYNC_BATCH / HYBRID execution patterns
- Agent-to-agent handoff workflows

**Key Features**:
- YAML/JSON workflow definitions
- Built-in HTTP connector for calling any REST API
- Connectors for 200+ GCP services (Cloud Batch, BigQuery, Pub/Sub, etc.)
- Parallel execution branches
- Sub-workflows for modular design
- Error handling with retry policies
- Execution history and logging
- Callbacks for long-running operations (wait for Cloud Batch job)

**Example -- Docking Pipeline**:
```yaml
main:
  steps:
    - submit_docking:
        call: googleapis.batch.v1.projects.locations.jobs.create
        args:
          parent: "projects/my-project/locations/us-central1"
          body:
            taskGroups:
              - taskSpec:
                  runnables:
                    - container:
                        imageUri: "us-central1-docker.pkg.dev/my-project/tools/diffdock:v1.1"
        result: job

    - wait_for_completion:
        call: googleapis.batch.v1.projects.locations.jobs.get
        args:
          name: ${job.name}
        result: job_status
        # Poll until SUCCEEDED or FAILED

    - store_results:
        call: googleapis.bigquery.v2.tabledata.insertAll
        args:
          projectId: "my-project"
          datasetId: "docking_results"
          tableId: "scores"
          body:
            rows: ${parse_results(job_status)}

    - notify:
        call: googleapis.pubsub.v1.projects.topics.publish
        args:
          topic: "projects/my-project/topics/pipeline-events"
          body:
            messages:
              - data: ${base64.encode(json.encode({"status": "complete", "job": job.name}))}
```

---

### 4.2 Pub/Sub

| Field | Value |
|-------|-------|
| **Service** | Cloud Pub/Sub |
| **API** | `pubsub.googleapis.com` |
| **Terraform** | `google_pubsub_topic`, `google_pubsub_subscription` |
| **Pricing** | $40/TB ingested (first 10GB/mo free) |
| **Status** | GA |

**Description**: Global, real-time messaging service for event-driven architectures. Exactly-once delivery, dead letter queues, and schema validation.

**Biotech Use Cases**:
- Agent-to-agent async messaging
- Pipeline event notifications (job complete, results ready)
- FHIR resource change notifications (via Healthcare API integration)
- Federated learning gradient aggregation (as in biotech-implementations/)
- Data ingestion triggers

**Key Features**:
- Push and pull subscriptions
- Exactly-once delivery (within a region)
- Message ordering (with ordering keys)
- Dead letter topics for failed messages
- Schema validation (Avro, Protocol Buffers)
- BigQuery subscriptions (auto-write messages to BigQuery)
- Message filtering (attribute-based)
- Up to 31 days message retention

---

### 4.3 Cloud Composer (Managed Apache Airflow)

| Field | Value |
|-------|-------|
| **Service** | Cloud Composer |
| **API** | `composer.googleapis.com` |
| **Terraform** | `google_composer_environment` |
| **Pricing** | Environment: $0.35-0.75/hr (Composer 2), plus Compute/GKE costs |
| **Status** | GA (Composer 2) |

**Description**: Managed Apache Airflow for authoring, scheduling, and monitoring complex data and ML pipelines as DAGs.

**Biotech Use Cases**:
- Complex multi-step bioinformatics pipelines
- Scheduled data ingestion (daily ClinVar updates, FAERS imports)
- ETL pipelines (raw sequencing -> processed -> BigQuery)
- ML training pipeline orchestration
- Regulatory report generation

**Key Features**:
- Composer 2: Built on GKE Autopilot, autoscaling workers
- Pre-built operators for GCP services (BigQuery, Dataflow, Cloud Batch, GCS)
- KubernetesPodOperator for custom containers
- Airflow 2.x with TaskFlow API
- Private IP environments for VPC-SC compliance
- Triggerer for efficient async sensing

---

### 4.4 Dataflow (Apache Beam)

| Field | Value |
|-------|-------|
| **Service** | Cloud Dataflow |
| **API** | `dataflow.googleapis.com` |
| **Terraform** | `google_dataflow_job` |
| **Pricing** | Per vCPU-hr, per GB-hr memory, per GB-hr storage |
| **Status** | GA |

**Description**: Fully managed stream and batch data processing service based on Apache Beam.

**Biotech Use Cases**:
- VCF -> BigQuery transformation (as in biotech-implementations/multiomnic-ref/)
- Real-time FHIR resource processing
- Streaming genomic data processing
- Large-scale data transformation and enrichment
- HL7v2 message parsing at scale

**Key Features**:
- Unified batch and streaming model
- Autoscaling workers
- Streaming Engine for low-latency processing
- FlexRS (Flexible Resource Scheduling) for cost-optimized batch
- Pre-built templates for common transforms
- GPU support for ML inference in pipelines

---

## 5. Security & Compliance Services

### 5.1 VPC Service Controls

| Field | Value |
|-------|-------|
| **Service** | VPC Service Controls |
| **API** | `accesscontextmanager.googleapis.com` |
| **Terraform** | `google_access_context_manager_service_perimeter` |
| **Pricing** | Free (no additional charge) |
| **Status** | GA |

**Description**: Creates security perimeters around GCP resources to prevent data exfiltration. Critical for healthcare/biotech data sovereignty.

**Biotech Use Cases**:
- Prevent genomic data from leaving a defined perimeter
- Multi-jurisdiction data sovereignty (US/EU/APAC)
- HIPAA-compliant data environments
- Federated learning with data isolation (as in Federated_Genomic/)

**Key Features**:
- Service perimeters restrict API access to defined projects
- Ingress/Egress rules for controlled cross-perimeter access
- Bridge perimeters for multi-project communication
- Supported services: BigQuery, GCS, Healthcare API, Vertex AI, Compute Engine, and 50+ more
- Dry-run mode for testing before enforcement
- Access levels based on IP, device, identity

**Protected Services Relevant to Biotech**:
- `bigquery.googleapis.com`
- `storage.googleapis.com`
- `healthcare.googleapis.com`
- `aiplatform.googleapis.com` (Vertex AI)
- `batch.googleapis.com`
- `compute.googleapis.com`
- `pubsub.googleapis.com`
- `dataflow.googleapis.com`

---

### 5.2 Cloud KMS

| Field | Value |
|-------|-------|
| **Service** | Cloud Key Management Service |
| **API** | `cloudkms.googleapis.com` |
| **Terraform** | `google_kms_key_ring`, `google_kms_crypto_key` |
| **Pricing** | Software keys: $0.06/10K ops; HSM keys: $1.00-2.50/key/mo + $0.03/10K ops |
| **Status** | GA |

**Description**: Managed encryption key service for customer-managed encryption keys (CMEK), key rotation, and HSM-backed keys.

**Biotech Use Cases**:
- CMEK encryption for genomic data in GCS and BigQuery
- CMEK for Healthcare API FHIR/DICOM stores
- Key rotation policies for compliance
- HSM-backed keys for highest security requirements
- EKM (External Key Manager) for BYOK scenarios

**CMEK Integration Matrix**:
| Service | CMEK Support |
|---------|-------------|
| BigQuery | Yes (dataset-level) |
| Cloud Storage | Yes (bucket-level) |
| Healthcare API | Yes (dataset-level) |
| Firestore | Yes (database-level) |
| Compute Engine / Cloud Batch | Yes (disk-level) |
| Vertex AI | Yes (training data, model artifacts) |
| Pub/Sub | Yes (topic-level) |

---

### 5.3 Cloud DLP (Data Loss Prevention)

| Field | Value |
|-------|-------|
| **Service** | Cloud Data Loss Prevention (Sensitive Data Protection) |
| **API** | `dlp.googleapis.com` |
| **Terraform** | `google_data_loss_prevention_inspect_template` |
| **Pricing** | $1.00-6.00 per GB inspected (volume discounts available) |
| **Status** | GA |

**Description**: Discover, classify, and protect sensitive data (PII, PHI, genomic identifiers) across GCP services.

**Biotech Use Cases**:
- PHI detection in clinical datasets before sharing
- PII scanning in research databases
- De-identification of patient data for secondary analysis
- Compliance auditing for HIPAA/GDPR

**Built-in InfoTypes Relevant to Biotech**:
| InfoType | Detects |
|----------|---------|
| `PERSON_NAME` | Patient names |
| `DATE_OF_BIRTH` | Birth dates |
| `MEDICAL_RECORD_NUMBER` | MRNs |
| `PHONE_NUMBER` | Contact numbers |
| `EMAIL_ADDRESS` | Email addresses |
| `US_SOCIAL_SECURITY_NUMBER` | SSNs |
| `GENERIC_ID` | Generic identifiers |
| Custom InfoTypes | Gene names, accession numbers, etc. (user-defined regex/dictionary) |

**Key Features**:
- Inspection (discover sensitive data)
- De-identification (redact, mask, tokenize, bucket)
- Re-identification (reverse tokenization with proper keys)
- Stored InfoType detectors (custom patterns)
- Integration with BigQuery, GCS, Healthcare API

---

### 5.4 Organization Policies

| Field | Value |
|-------|-------|
| **Service** | Organization Policy Service |
| **API** | `orgpolicy.googleapis.com` |
| **Terraform** | `google_org_policy_policy` |
| **Pricing** | Free |
| **Status** | GA |

**Description**: Centralized constraints on GCP resource configurations at the organization, folder, or project level.

**Key Policies for Biotech**:
| Policy | Constraint ID | Purpose |
|--------|--------------|---------|
| Resource location restriction | `constraints/gcp.resourceLocations` | Enforce data residency (US-only, EU-only) |
| Disable serial port access | `constraints/compute.disableSerialPortAccess` | Security hardening |
| Require OS Login | `constraints/compute.requireOsLogin` | Enforce managed SSH |
| Restrict VPC peering | `constraints/compute.restrictVpcPeering` | Network isolation |
| Uniform bucket-level access | `constraints/storage.uniformBucketLevelAccess` | Consistent IAM |

---

### 5.5 Cloud Audit Logs

| Field | Value |
|-------|-------|
| **Service** | Cloud Audit Logs |
| **API** | Built into all GCP services |
| **Pricing** | Admin Activity: Free; Data Access: charged at Cloud Logging rates |
| **Status** | GA |

**Description**: Immutable audit trail for all GCP API calls. Critical for regulatory compliance.

**Log Types**:
| Type | What It Records | Biotech Relevance |
|------|----------------|-------------------|
| Admin Activity | Resource creation/deletion/modification | Infrastructure changes audit |
| Data Access | Data read/list operations | Who accessed patient data |
| System Event | GCP-initiated changes | Automatic scaling events |
| Policy Denied | Access attempts blocked by IAM/VPC-SC | Unauthorized access detection |

---

### 5.6 Compliance Certifications

Google Cloud maintains the following compliance certifications relevant to biotech:

| Certification | Relevance | Notes |
|--------------|-----------|-------|
| **HIPAA BAA** | Protected Health Information | Available for 100+ GCP services. Requires BAA with Google. |
| **HITRUST CSF** | Healthcare security framework | Certified for core services |
| **SOC 1/2/3** | Security, availability, confidentiality | All services |
| **ISO 27001** | Information security management | All services |
| **ISO 27017** | Cloud security controls | All services |
| **ISO 27018** | PII protection in cloud | All services |
| **FedRAMP High** | US federal government | Select services |
| **GDPR** | EU data protection | Supported with Data Processing Addendum |
| **GxP** | Good manufacturing/lab/clinical practice | Supported via compliance guides (not a certification per se) |

---

## 6. Additional Services

### 6.1 Artifact Registry

| Field | Value |
|-------|-------|
| **Service** | Artifact Registry |
| **API** | `artifactregistry.googleapis.com` |
| **Terraform** | `google_artifact_registry_repository` |
| **Pricing** | Storage: $0.10/GB/mo; Network: standard egress |
| **Status** | GA |

**Description**: Managed container and package registry. Stores Docker images, Python packages, Maven/npm packages.

**Biotech Use Cases**:
- Store agent Docker images (DiffDock, GROMACS, DeepVariant containers)
- Private Python package registry for internal scientific libraries
- Vulnerability scanning for containers
- Integration with Cloud Build CI/CD

---

### 6.2 Secret Manager

| Field | Value |
|-------|-------|
| **Service** | Secret Manager |
| **API** | `secretmanager.googleapis.com` |
| **Terraform** | `google_secret_manager_secret` |
| **Pricing** | $0.06 per 10K access operations; $0.03 per active secret version/mo |
| **Status** | GA |

**Description**: Store and manage API keys, passwords, certificates, and other sensitive data.

**Biotech Use Cases**:
- API keys for external databases (PubChem, UniProt, ClinicalTrials.gov)
- Service account keys
- Database credentials
- Encryption passphrases

---

### 6.3 Cloud Logging & Monitoring

| Field | Value |
|-------|-------|
| **Service** | Cloud Logging / Cloud Monitoring / Cloud Trace |
| **API** | `logging.googleapis.com`, `monitoring.googleapis.com`, `cloudtrace.googleapis.com` |
| **Pricing** | First 50 GB/mo free (Logging); Monitoring: free for GCP metrics |
| **Status** | GA |

**Description**: Observability stack for logs, metrics, traces, and dashboards.

**Biotech Use Cases**:
- Agent execution tracing (end-to-end latency)
- Pipeline monitoring dashboards
- Cost anomaly alerting
- Error reporting for batch jobs
- Custom metrics for scientific workflows (compounds processed, variants called)

---

### 6.4 Filestore

| Field | Value |
|-------|-------|
| **Service** | Cloud Filestore |
| **API** | `file.googleapis.com` |
| **Terraform** | `google_filestore_instance` |
| **Pricing** | Basic HDD: $0.20/GB/mo; Basic SSD: $0.27/GB/mo; Enterprise: $0.36/GB/mo |
| **Status** | GA |

**Description**: Managed NFS file server for workloads requiring shared filesystem access.

**Biotech Use Cases**:
- Shared filesystem for GKE-based scientific clusters
- AlphaFold genetic database hosting (2.2TB, needs fast random read)
- Shared model weights across multiple batch jobs
- Nextflow work directories

---

## 7. Service-to-Service Integration Matrix

This matrix shows how services connect in typical biotech pipelines:

```
                    Cloud    Cloud   Big    Health-  Cloud    Pub/    Cloud    Vertex   Fire-
                    Batch    Run     Query  care API Storage  Sub     Workflows AI       store
Cloud Batch         -        -       Write  -        R/W      Pub     Trigger  -        -
Cloud Run           -        -       R/W    R/W      R/W      Pub/Sub Trigger  API      R/W
BigQuery            -        -       -      Import   Import   Sub     -        ML       -
Healthcare API      -        -       Export -        -        Pub     -        -        -
Cloud Storage       Mount    Mount   Load   Import   -        -       -        Train    -
Pub/Sub             -        Trigger -      Notify   -        -       Trigger  -        -
Cloud Workflows     Submit   Call    Query  -        R/W      Pub     -        Call     -
Vertex AI           -        -       Train  -        R/W      -       -        -        -
Firestore           -        R/W     -      -        -        -       -        -        -
```

---

## 8. Pricing Summary for Common Biotech Workloads

| Workload | Services | Estimated Cost | Notes |
|----------|----------|---------------|-------|
| **Variant Calling (30x WGS)** | Cloud Batch (CPU) + GCS + BigQuery | $0.50-5.00/genome | Spot VMs, DeepVariant |
| **Molecular Docking (10K compounds)** | Cloud Batch (L4 GPU) + GCS | $10-50/batch | DiffDock, Spot VMs |
| **AlphaFold Prediction (single protein)** | Cloud Batch (A100) + GCS | $5-50/prediction | Depends on protein length |
| **GROMACS MD (100ns)** | Cloud Batch (A100) + GCS | $20-100/simulation | GPU hours vary |
| **BigQuery Analytics (1TB scan)** | BigQuery | $6.25 | On-demand pricing |
| **Agent Hosting (idle)** | Cloud Run (scale-to-zero) | ~$0/month | When not receiving requests |
| **Agent Hosting (active)** | Cloud Run | $5-50/month | Depends on traffic |
| **FHIR Store (10K patients)** | Healthcare API | $10-30/month | Storage + operations |
| **Inventory Vector Search** | Vertex AI Vector Search | ~$90/month | 1 shard, always-on |
| **Compound Screening Pipeline** | Batch + Run + BQ + GCS + Workflows | $50-200/run | End-to-end pipeline |

---

## 9. Region Availability

Recommended regions for biotech workloads:

| Region | Location | GPU Availability | Healthcare API | Why |
|--------|----------|-----------------|---------------|-----|
| `us-central1` | Iowa | T4, L4, A100, H100 | Yes | Best GPU availability, most services |
| `us-east4` | Virginia | T4, L4, A100 | Yes | East coast, low latency to NIH/FDA |
| `europe-west4` | Netherlands | T4, L4, A100 | Yes | EU data residency (GDPR) |
| `europe-west1` | Belgium | T4, L4 | Yes | EU alternative |
| `asia-southeast1` | Singapore | T4, L4 | Yes | APAC data residency |

**Recommendation**: Default to `us-central1` for best GPU availability and service coverage. Use multi-region for data sovereignty requirements.

---

## 10. Inventory Bootstrap Asset IDs

These are the recommended `asset_id` values for the Day 1 Content Pack:

### Services
```
inv-svc-cloud-batch
inv-svc-cloud-run
inv-svc-gke
inv-svc-compute-engine
inv-svc-vertex-ai
inv-svc-agent-engine
inv-svc-vector-search
inv-svc-bigquery
inv-svc-healthcare-api
inv-svc-gcs
inv-svc-firestore
inv-svc-cloud-workflows
inv-svc-pubsub
inv-svc-cloud-composer
inv-svc-dataflow
inv-svc-vpc-sc
inv-svc-cloud-kms
inv-svc-dlp
inv-svc-artifact-registry
inv-svc-secret-manager
inv-svc-filestore
inv-svc-cloud-logging
inv-svc-cloud-trace
```

### Models
```
inv-model-gemini-3-flash
inv-model-gemini-3-pro
inv-model-gemini-1-5-pro
inv-model-medgemma
inv-model-txgemma
inv-model-med-palm-2
inv-model-alphafold
inv-model-deepvariant
inv-model-esm2
inv-model-text-embedding-005
```

### Datasets
```
inv-dataset-1000-genomes
inv-dataset-gnomad
inv-dataset-clinvar
inv-dataset-ncbi-gene
inv-dataset-fda-faers
inv-dataset-fda-drug-label
inv-dataset-google-patents
inv-dataset-alphafold-db
inv-dataset-covid19
inv-dataset-who-health
inv-dataset-gcs-genomics-public
inv-dataset-gcs-gnomad
inv-dataset-gcs-broad-references
inv-dataset-gcs-alphafold-structures
inv-dataset-gcs-gatk-best-practices
```

### Templates
```
inv-template-drug-screening-pipeline
inv-template-variant-calling-pipeline
inv-template-protein-engineering-pipeline
inv-template-literature-analysis-pipeline
inv-template-adverse-event-surveillance
inv-template-clinical-genomics-report
```
