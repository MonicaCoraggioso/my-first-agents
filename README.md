# my-first-agents

Learning to build AI agents from scratch with a small open-source model (MiniCPM) running locally.

## `projects/` — my own work

| File | What it is |
|---|---|
| [`1-physics_agent_pendulum.ipynb`](projects/1-physics_agent_pendulum.ipynb) | A **physics agent**: the agent loop from the course applied to a physics problem. Its numerical tools simulate the nonlinear pendulum (`solve_ivp`), measure the period and compute the small-angle prediction; the agent decides which simulations to run and explains why the period grows with amplitude. Tools are verified against the exact elliptic-integral solution. Includes an analysis of a real run and exercises. |
| [`2-physics_agent_pendulum_GAME.ipynb`](projects/2-physics_agent_pendulum_GAME.ipynb) | The same physics agent rebuilt with the **GAME framework** from the course: physics tools as `Action`s, instructions as `Goal`s, and the framework loop unchanged. Adds what notebook 1 lacked: a tolerant parser and a retry with an actionable error message, plus a `SafeAgent` subclass so a bad tool call never crashes the loop. Task: find the amplitude at which the period is 10% longer than the small-angle period; the agent's simulations are checked against the exact solution. |

## `course/` — following the Coursera course on agentic AI

Based on [*AI Agents and Agentic AI with Python & Generative AI*](https://www.coursera.org/learn/ai-agents-python) (Vanderbilt University, Coursera). Notebooks are numbered in the order of the course and use MiniCPM locally instead of the course's hosted models. See [`course/README.md`](course/README.md) for details.

| File | What it is |
|---|---|
| [`0-test_MiniCPM.ipynb`](course/0-test_MiniCPM.ipynb) | First experiment: loads **MiniCPM** (`openbmb/MiniCPM5-2B`) with 🤗 `transformers`, defines a `generate_response()` helper for chat messages, and tests a system-prompt fact and a JSON-structured coding request. |
| [`1-building_quasi_agent.ipynb`](course/1-building_quasi_agent.ipynb) | A **quasi-agent**: chains three prompts with a shared conversation memory to turn a one-line description into a function, add its documentation, write `unittest` tests, and save everything to a `.py` file. |
| [`2-building_my_first_simple_agent.ipynb`](course/2-building_my_first_simple_agent.ipynb) | A first real **agent** that can list and read files and answer questions about them. Builds the full **agent loop** step by step: construct prompt (rules + memory) → generate response → parse `{tool_name, args}` (with error fallback) → execute the tool → update memory → continue or stop (`terminate` / max iterations), then runs it all in one `while` loop with improved agent rules (JSON tool schema + step-by-step thinking). |
| [`2-agent_loop_notes.md`](course/2-agent_loop_notes.md) | Notes for notebook 2: the six steps of the agent loop, agent rules, parsing with error fallback, execution, memory, termination, improved rules (GAIL), and lessons from running it with MiniCPM. |
| [`3-building_agent_with_GAME.ipynb`](course/3-building_agent_with_GAME.ipynb) | A small reusable **agent framework** designed with **GAME** (Goals, Actions, Memory, Environment): `Goal`, `Action` + `ActionRegistry`, `Memory`, `Environment`, an `AgentLanguage` that builds the prompt and parses the reply, and an `Agent` class running the loop. Uses MiniCPM with prompt-based function calling (tolerant parser + retry). |
| [`3-GAME_framework_notes.md`](course/3-GAME_framework_notes.md) | Notes for notebook 3: simulating agents in a chat, the GAME components, the agent loop, `AgentLanguage`, tool decorators and tag-based registries, plus how the notebook differs from the course. |

## Other files

| File | What it is |
|---|---|
| `requirements.txt` | Python packages for the local environment. |
| `.gitignore` | Files Git should ignore (caches, virtual environments, `.DS_Store`, …). |
