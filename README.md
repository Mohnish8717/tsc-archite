# Predictive Reality Engine
### *Autonomous Social Simulation & Adversarial Boardroom Debate*

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![AG2](https://img.shields.io/badge/AG2-(AutoGen)-orange?style=for-the-badge)](https://microsoft.github.io/autogen/)
[![CAMEL OASIS](https://img.shields.io/badge/Simulation-OASIS-purple?style=for-the-badge)](https://github.com/camel-ai/oasis)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

**Find out if your feature will fail before you spend 3 months building it.**

[Overview](#overview) • [Live Demos](#live-simulation-examples) • [How it Works](#the-8-step-process) • [Quick Start](#how-to-run-it) • [Runtime Modes](#runtime-modes) • [Community](#join-the-conversation)

</div>

---

## Overview

Right now, building software is mostly guesswork. We come up with an idea, build it, ship it, and just hope people actually want it. If they don't, we've just burned a ton of time, money, and engineering effort.

**Predictive Reality Engine** fixes that. It's a simulation engine that lets you see how the market will react *before* you write any code.

We spin up a synthetic social network filled with AI users based on your actual customer data. We drop your feature idea into the mix and watch them react. After that, an AI boardroom—featuring a CTO, CFO, CISO, and others—takes that user backlash and aggressively debates whether the feature is actually worth building given your real-world constraints.

*   **For Leadership (Macro):** It's a safe sandbox to test technical debt and ROI without risking actual money or engineer hours.
*   **For Product Teams (Micro):** It flags onboarding friction, edge cases, and user hostility instantly.

Basically, we let you play out the future of your product so you don't build the wrong thing.

---

## Live Simulation Examples

Want to see what this actually looks like? Here is a scenario we've already run:

### 1. The AI Fitness Platform
Watch our simulated market tear apart the onboarding flow of a proposed AI Fitness app. The backlash forces the Autonomous Boardroom (CTO, CISO, CFO) into a cut-throat debate over edge processing, HIPAA compliance, and 60fps latency targets.
*(Demo videos and interactive views coming soon...)*

---

## The 8-Step Process

Here is exactly how the engine works from start to finish:

1. **Ingest Real Data:** We pull in your raw Zendesk tickets, Slack logs, and user interviews to figure out what's actually broken.
2. **Find the Pain:** The system clusters that data to find exactly what users hate, and automatically drafts a Feature Proposal to fix it.
3. **Build the Personas:** We generate AI agents that act exactly like your real users—complete with their specific biases and habits.
4. **Assemble the Boardroom:** We set up an AI executive suite (CEO, CTO, CISO, CMO, CFO, CPO, Legal, Data, Sales, CS). Each one is briefed with real company constraints (e.g., the CFO knows we're running out of cash).
5. **Run the Simulation:** We drop 'Seeds'—initial controversial takes about the feature—into our OASIS social network. Hundreds of AI users react, argue, and complain in real-time.
6. **The Debate:** The executives take that backlash and fight over it. There are no yes-men here. If the CTO says the feature is too slow, they have to back it up with hard data from the **World RAG**.
7. **Generate the Spec (WIP):** If the feature survives the boardroom, we automatically turn the consensus into a ready-to-build PRD.
8. **Handoff:** We output the final engineering tickets, integration tests, and monitoring plans so your team can just start building.

---

## How to Run It

### What you need
Make sure you have these installed:
*   `python 3.10+`
*   `node 18+` (for the UI)
*   `docker` & `docker-compose` (for the memory engine)

### Setup (The Full Stack)

#### 1. API Keys
```bash
# Copy the example config
cp .env.example .env

# Open .env and drop in your GROQ_API_KEY and OPENAI_API_KEY
```

#### 2. Start the World RAG
The engine needs a memory bank to store everything. Boot up the local Docker setup:
```bash
chmod +x start_hindsight_local.sh
./start_hindsight_local.sh start
```
*   **API:** http://localhost:8888
*   **Control Plane:** http://localhost:9999

#### 3. Install Packages
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd predictive_ui && npm install && cd ..
```

#### 4. Run Everything
We made a quick script that boots up the Uvicorn backend, the WebSocket server, and the Vite frontend all at once:
```bash
./start_dev.sh
```
*   **Frontend UI:** http://localhost:5173
*   **Backend API:** http://localhost:8000

*(Note: If you just want to run headless simulations in the terminal, you can just run `./run_oasis.sh tsc/scripts/run_ai_fitness_case_study.py` instead).*

---

## Runtime Modes

Depending on what you're trying to do, you can run the engine in a few different ways:

| Mode | Command | What it does |
| :--- | :--- | :--- |
| **Standard** | `./run_oasis.sh tsc/scripts/run_ai_fitness_case_study.py` | Runs a standard case study with the full boardroom debate. |
| **Production** | `python run_production_pipeline.py` | The real deal. Hook it up to your actual enterprise data. |
| **Simulation Only** | `./run_oasis.sh` | Just runs the social simulation. Skips the boardroom debate entirely. |
| **Purge Memory** | `./start_hindsight_local.sh purge` | Wipes the simulation memory clean so you can start fresh. |
| **Hard Reset** | `./clean_start.sh` | Deletes all temporary logs, databases, and artifacts. |

---

## Built-in Checks

We put a few guardrails in place to make sure the output isn't garbage:

| Check | What it does |
| :--- | :--- |
| **Market Fit Gate** | Makes sure the AI personas actually match the real user data we fed into Layer 1. |
| **Red Team Gate** | Tries to actively break the simulation or find fatal security flaws (like privacy leaks) before the executives even see it. |
| **Logic Validator** | Makes sure the final PRD actually fixes the problems the boardroom complained about. |

---

## What You Get Out Of It

At the end of every run, you get a full Prediction Report:
*   **Market Sentiment:** Charts showing how user trust and frustration changed over time.
*   **Decision Journals:** A timeline of exactly when and why the market turned on your feature.
*   **Tension Ledger:** A paper trail explaining exactly why an executive voted to kill the project.

---

## Troubleshooting

*   **App crashing on Mac?** Apple Silicon sometimes deadlocks with PyTorch. We built a "Darwin Immunity" fix for this, just make sure you have `nest_asyncio` installed.
*   **Engine failing at Layer 6?** The boardroom probably can't talk to the memory bank. Double check that Hindsight is running by typing `./start_hindsight_local.sh status`.

---

## Join the Conversation
Building software blindly is a massive waste of time. If you're interested in using AI agents to simulate the market and stress-test product ideas, we'd love to chat. Join the community to talk about autonomous boardrooms and the future of product development.

## Shoutouts
A huge thanks to the CAMEL-AI team. The social simulation layer of this engine relies heavily on their open-source [OASIS](https://github.com/camel-ai/oasis) framework. 

<div align="center">
  <br>
  <i>"The future belongs to those who can simulate it."</i>
  <br>
  <b>Built with ❤️ </b>
</div>
