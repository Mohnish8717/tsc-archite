# 🔮 Predictive Reality Engine
### *Autonomous Social Simulation & Adversarial Boardroom Debate*

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![AG2](https://img.shields.io/badge/AG2-(AutoGen)-orange?style=for-the-badge)](https://microsoft.github.io/autogen/)
[![CAMEL OASIS](https://img.shields.io/badge/Simulation-OASIS-purple?style=for-the-badge)](https://github.com/camel-ai/oasis)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

**Validate product-market fit, predict regulatory friction, and generate technical specifications before writing a single line of code.**

[Overview](#-overview) • [Architecture](#-how-it-works-the-8-layer-stack) • [Quick Start](#-supercharged-quick-start) • [Runtime Modes](#-runtime-modes) • [Quality Gates](#-built-in-quality-gates) • [Telemetry](#-analytics--telemetry)

</div>

---

## 🌟 Overview

Modern software development is plagued by a massive blind spot: we build features based on hunches, deploy them, and only *then* discover how the market reacts. 

**The Predictive Reality Engine** fundamentally inverses this paradigm. It is an end-to-end **Autonomous Software Factory** that allows you to simulate the future of your product. By combining massive-scale synthetic user simulation with a highly adversarial, agentic boardroom debate, the system stress-tests feature proposals against technical debt, financial budgets, and user backlash—all in a completely synthetic, high-fidelity environment.

---

## 🏗️ How It Works: The 8-Layer Stack

The pipeline is a highly orchestrated flow of autonomous intelligence, shifting from raw data ingestion to social simulation, and finally to executive consensus.

### 🏛️ Layer 1: Contextual Ingestion
Ingests raw enterprise data (Zendesk tickets, Slack logs, customer interviews) and extracts core semantic signals using RAG-enhanced parsers.

### 🔍 Layer 2: Feature Discovery
Dynamically clusters pain points into "Tension Clusters" and automatically drafts compelling **Feature Proposals** that address real customer friction.

### 👥 Layer 3: Persona Generation
Builds deep psychological **User Personas** representing your actual market segments. These aren't just "profiles"—they are agents with cognitive biases, social interaction patterns, and distinct value systems.

### 👔 Layer 4: Boardroom Assembly
Initializes the **Autonomous Executive Suite** (CEO, CTO, CISO, Legal, Product). Each executive is pre-warmed with company-specific context, budget constraints, and "Private Intelligence" briefs (e.g., the CFO knows the *actual* runway, which may differ from public reports).

### 🌊 Layer 5: OASIS Market Simulation
Spins up hundreds of synthetic users in a simulated social media environment (CAMEL-AI OASIS). They interact, argue, post, and comment on the proposal, generating high-fidelity behavioral data that predicts market adoption and churn.

### ⚔️ Layer 6: AG2 Adversarial Debate
Executives debate the feature, grounded in simulation data.
*   **Zero Hallucination:** Agents query the **Hindsight Memory Bank** for actual user comments and metrics generated in Layer 5.
*   **Anti-Sycophancy:** Built-in logit-bias manipulation forces agents to challenge each other, preventing "echo chamber" consensus.

### 📝 Layer 7: Spec Generation
If the feature survives the boardroom, the system automatically compiles the debate consensus into a high-fidelity **PRD**, generating UI changes, data model updates, and prioritized development tasks.

### 📦 Layer 8: Technical Handoff
Generates the final engineering-ready artifacts, including integration tests and monitoring plans, for direct deployment into development environments.

---

## 🛡️ Built-in Quality Gates

The pipeline includes rigorous "Gates" that verify artifacts between layers to prevent garbage-in, garbage-out.

| Gate | Name | Description |
| :--- | :--- | :--- |
| **4.5** | **Market Fit Gate** | Evaluates if the generated personas actually represent the customer feedback ingested in Layer 1. |
| **4.6** | **Red Team Gate** | An adversarial agent attempts to "poison" the simulation or find fatal technical flaws (RCE, privacy leaks) before the boardroom convenes. |
| **7.2** | **Logic Validator** | Ensures the PRD generated in Layer 7 strictly follows the boardroom's mitigation requirements. |

---

## ⚡ Supercharged Quick Start

Follow these steps to get the Predictive Reality Engine running in less than 5 minutes.

### 1️⃣ Environment Setup
Clone the repo and install the core dependencies:
```bash
git clone https://github.com/your-org/tsc-architecture.git
cd tsc-architecture
pip install -r requirements.txt
```

### 2️⃣ Start Hindsight (The Memory Engine)
The system requires a persistent memory bank. Launch the self-hosted Hindsight server using Docker:
```bash
chmod +x start_hindsight_local.sh
./start_hindsight_local.sh start
```
*   **API Server:** [http://localhost:8888](http://localhost:8888)
*   **Control Plane:** [http://localhost:9999](http://localhost:9999)

### 3️⃣ Configure API Keys
Copy `.env.example` to `.env` and add your LLM provider keys:
```env
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
HINDSIGHT_URL=http://localhost:8888
```

### 4️⃣ Launch Your First Simulation
Run the Slack AI Controversy scenario:
```bash
python run_slack_scenario.py
```

---

## 🔄 Runtime Modes

The engine supports multiple execution modes depending on your research needs:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Standard Scenario** | `python run_slack_scenario.py` | Runs a pre-configured scenario with 20 simulations and full boardroom debate. |
| **Production Pipeline** | `python run_production_pipeline.py` | High-stakes mode for real enterprise data. Supports parallel simulation scaling. |
| **OASIS Isolated** | `./run_oasis.sh` | Runs only the social simulation layer to test user sentiment without executive debate. |
| **Memory Purge** | `./start_hindsight_local.sh purge` | Deletes all simulation memory banks for a fresh start. |
| **Clean Reset** | `./clean_start.sh` | Purges all temporary logs, databases, and local artifacts. |

---

## 📊 Analytics & Telemetry

The engine generates a comprehensive **Prediction Report** at the end of every run:

*   **Market Sentiment Series:** Time-series data of user trust and frustration (NPS/Churn).
*   **Decision Journals:** A chronological log of every cognitive threshold crossed by the simulated market.
*   **Tension Ledger:** An audit trail of why the CTO voted "High Risk."
*   **PRD Output:** Found in `slack_scenario_result.json` or your specified output path.

---

## 🛠️ Advanced Troubleshooting

*   **macOS Deadlock:** The system includes a "Darwin Immunity" layer to prevent gRPC/PyTorch deadlocks common on Apple Silicon. Ensure `nest_asyncio` is installed.
*   **Hindsight Unreachable:** If Layer 6 fails, verify Hindsight is running with `./start_hindsight_local.sh status`.
*   **Memory Issues:** Simulations with >50 agents require at least 16GB RAM. For 1000+ agents, use a cloud-hosted Hindsight cluster.

---

<div align="center">
  <i>"The future belongs to those who can simulate it."</i>
  <br>
  <b>Built with ❤️ </b>
</div>
