# MiroFish — System Architecture: Enterprise-Grade Social Simulation

MiroFish is a multi-agent decision-support platform designed to evaluate feature proposals by simulating their impact on a virtual market of grounded, evidence-backed personas. It employs a sophisticated 8-layer pipeline that transforms raw qualitative data into actionable, risk-aware technical specifications.

---

## 1. System Philosophy: Grounded Intelligence
Unlike traditional RAG systems that perform simple vector lookups, MiroFish enforces **strict evidentiary grounding**:
- **Facts over Hallucinations**: Every agent trait or simulation parameter must cite a source snippet (quote/fact) from the original documents.
- **Adversarial Pressure**: Stakeholders do not just "collaborate"—they are programmed to hunt for risks and critique internal inconsistencies.
- **Stochastic Realism**: Market adoption is modeled using Monte Carlo simulations (Mesa), not just static LLM predictions.

---

## 2. Core Data Models (The Payload)
- **`FeatureProposal`**: The central object under evaluation. Contains title, technical description, target users, and constraints.
- **`ProblemContextBundle`**: The output of Layer 1. A collection of semantically grouped document chunks, extracted entities, and urgency metrics.
- **`KnowledgeGraph`**: A relational map of entities (Users, Systems, Organizations) stored in Zep Cloud.
- **`FinalPersona`**: A high-fidelity agent profile including MBTI, drivers, and the `CitedEvidence` that justifies their existence.
- **`GatesSummary`**: An aggregate scorecard of the 8 evaluation gates, containing verdicts, scores, and risk mitigations.
- **`FinalRecommendation`**: The end-state product of the pipeline, including the final verdict (APPROVE/REJECT), a risk-weighted spec, and a leadership summary.

---

## 3. The 8-Layer Pipeline Architecture

### Layer 1: Contextual Ingestor (The "Sensor")
- **Recursive Semantic Chunking**: Documents are split into segments that maintain semantic coherence, avoiding mid-sentence cuts that destroy context.
- **FastEmbed Integration**: Uses local ONNX-accelerated vectors (`BAAI/bge-small-en-v1.5`) for high-speed embedding and macOS stability.
- **Entity & Sentiment Extraction**: Identifies technical mentions, sentiment (positive/negative), and urgency (1-5) to guide downstream prioritization.

### Layer 2: Knowledge Graph Builder (The "Memory")
- **Zep Cloud Fact Store**: Ingests document facts with explicit mention metadata (text snippets, source IDs).
- **Relational Mapping**: Constructs a NetworkX-based graph to identify dependencies (e.g., "Feature A requires Infrastructure B").

### Layer 3: Grounded Persona Generator (The "Agents")
- **Stakeholder Selection**: Dynamically identifies relevant internal and external personas based on the feature proposal (e.g., Privacy Lead for data-heavy features).
- **Evidence-First Construction**: Uses a specialized "Grounded" prompt that forces the LLM to build psychological profiles exclusively from cited facts in the KG.
- **Validation**: Every profile undergoes a grounding check; traits without verifiable citations are flagged or removed.

### Layer 4: Gate Evaluation (The "Filter")
The system runs 8 specialized "Gates" using Llama-3.1 (Groq):
1.  **Technical Code Viability**: Logic/Architecture feasibility.
2.  **SOTA Probe**: Market uniqueness and novelty.
3.  **Resource Impact**: Team bandwidth and budget assessment.
4.  **Infrastructure Requirements**: CDN, Database, and API dependencies.
5.  **Market Fit (Mesa)**: 300+ agent Monte Carlo simulation of user adoption curves.
6.  **Red-Team (Adversarial)**: Simulation of failure modes (e.g., security breach, sudden churn).
7.  **Feature Interactions**: Conflicts with existing products.
8.  **Execution Feasibility**: Timeline, budget, and regulatory compliance.

### Layer 5: Iterative Refinement (The "Optimizer")
- If the Gate score falls below a "FAIL" threshold, Layer 5 uses a refinement engine to suggest modifications to the feature proposal (e.g., "Simplify the MVP to reduce infrastructure debt").

### Layer 6: Adversarial Stakeholder Debate (The "Consensus")
Stakeholders engage in a 3-round adversarial cycle:
- **Round 1: Initial Position**: Stakeholders state their stance with evidence-based reasoning.
- **Round 2: Adversarial Critique**: Stakeholders are instructed to hunt for flaws in others' arguments and pressure-test their assumptions.
- **Round 3: Rebuttal & Consensus**: Final synthesis of tensions into a unified recommendation (APPROVE, REJECT, or CONDITIONALLY APPROVE).

### Layer 7: Specification Generation (The "Blueprint")
- Converts the refined proposal and debate insights into a technical PRD.
- Generates a **Task Table** (P0-P2) with effort estimates and dependency tracking.

### Layer 8: Handoff & Report (The "Executive Summary")
- Packages the final recommendation for delivery to senior leadership and developers.

---

## 4. Technology Stack & Infrastructure
- **LLM Orchestration**: Groq (Llama-3.1-8b/70b) with `asyncio` parallel execution and robust rate limiting.
- **Memory**: Zep Cloud (Long-term Fact/Graph Store).
- **Simulation**: Mesa (Agent-Based Modeling) + Scipy (Adoption statistics).
- **Embeddings**: FastEmbed (Local, ONNX-accelerated).
- **Database**: PostgreSQL (Structured data persistence).
- **Caching**: Redis/LRU Cache (LLM and Persona deduplication).

---
*MiroFish: Grounded social simulation for evidence-based decision making.*
