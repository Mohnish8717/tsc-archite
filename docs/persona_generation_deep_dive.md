# Layer 3 Deep Dive: Grounded Persona Generator

Layer 3 is the "Social Core" of MiroFish. It transforms raw evidentiary facts and graph entities into high-fidelity, psychologically consistent personas that act as agents in downstream simulations (Gates and Debate).

---

## 🏗️ 1. Dynamic Stakeholder Selection
The system does not use static personas. Instead, it dynamically identifies the most relevant stakeholders for each `FeatureProposal`.

### Internal Stakeholders
- **Logic**: Analyzes `feature.affected_domains` and `feature.description` against the top 15 entities in the Knowledge Graph.
- **Selection**: Uses a specialized LLM prompt (`STAKEHOLDER_SELECTION_USER`) to rank organizational roles based on relevance.
- **Output**: 3-5 roles (e.g., "Privacy Lead", "Infrastructure Architect") with assigned decision authority and domain expertise.

### External Customer Personas
- **Logic**: Analyzes `target_users` and `target_user_count` to segment the market.
- **Categorization**: Maps personas into segments like "Power User," "Routine User," or "Enterprise Buyer."
- **Relevance**: Stakeholders are scored 0-1 based on how directly the feature solves their documented pain points.

---

## 📝 2. Evidence-First Profile Construction
The "Grounded" persona architecture moves beyond simple role-playing by forcing the LLM to build profiles from specific, categorized facts.

### The Grounded Prompt Builder
The `_build_grounded_persona_prompt` method assembles a context bundle with four distinct evidence layers:
1.  **Direct Quotes (Primary Evidence)**: Exact snippets from the stakeholder's interviews/tickets. This defines their "voice."
2.  **Observational Evidence**: Categorized into *Drivers* (positive sentiment) and *Pain Points* (negative sentiment/urgency).
3.  **Organizational Context**: The company's tech stack, budget, and constraints.
4.  **Constraints & Urgency**: High-importance entities from the KG that this stakeholder specifically cares about.

### Psychological Modeling
The system uses the **MBTI** framework as a baseline for cognitive style, but maps it to behavior using document evidence:
- **Decision Patterns**: Deduced from facts about speed and data preference.
- **Communication Style**: Based on the tone of direct quotes (e.g., "Direct" vs "Diplomatic").

---

## ✅ 3. Validation & Grounding Verification
To prevent LLM hallucination, MiroFish employs a multi-step validation engine.

### Regex-Based Citation Check (`_validate_grounding`)
- **Mechanism**: The LLM is instructed to use a `[Fact: <snippet>]` notation for every claim.
- **Verification**: The system uses regex to extract all citations and cross-references them against the original `StakeholderContextBundle`.
- **Scoring**: A `grounding_score` (0-1) is calculated based on the ratio of valid citations to total claims.

### Evidence-Based Confidence Calculation
The `_estimate_profile_confidence` method calculates a final reliability score based on:
- **Context Depth**: +0.3 for 10+ facts.
- **Profile Depth**: +0.25 for 2000+ words of generated text.
- **Structural Integrity**: +0.15 for presence of MBTI and key behavioral sections.

---

## 📊 Summary of Implementation
| Component | Technique | Purpose |
| :--- | :--- | :--- |
| **Selection** | Semantic Matching | Ensures the right stakeholders evaluate the feature. |
| **Construction** | Evidentiary Prompting | Forces LLM to stay within the bounds of document facts. |
| **Parsing** | Keyword/Regex Extraction | Translates free-form LLM text into structured Pydantic models. |
| **Verification** | Citation Cross-Reffing | Detects and penalties hallucinations (score < 0.3 = Risky). |
