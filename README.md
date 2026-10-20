# my-first-agents

Learning to build AI agents from scratch with a small open-source model (MiniCPM) running locally.

## `projects/` — my own work

| File | What it is |
|---|---|
| [`1-lab_data_agent_unknown_planet.ipynb`](projects/1-lab_data_agent_unknown_planet.ipynb) | A **lab-data agent** that measures *g* on an "unknown planet". Pendulum measurements are generated with a random, hidden *g*, noise and damping, and saved as messy CSV files (a length in cm, a large-amplitude run, a corrupted file). The agent (a small local LLM in a tool-using loop) explores the files, measures the periods, decides which runs to trust, fits *g* and explains its choices. **Saved run:** *g* within 0.1% of the hidden value, all three data problems caught; the same LLM without tools could not answer. Uses `agent_framework.py`; explanations included. |
| [`agent_framework.py`](projects/agent_framework.py) | A small reusable **agent framework** (Goals, Actions, Memory, Environment), adapted from the course for a local LLM: emulated function calling with a tolerant parser and retry, and a loop that never crashes on bad tool calls. Imported by the notebooks in `projects/`. |

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
