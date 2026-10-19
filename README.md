# my-first-agents

A sandbox for learning to build AI agents with small open-source language models, starting with running a chat model locally or in Google Colab.

## Contents

| File | What it is |
|---|---|
| [`0-test_MiniCPM.ipynb`](0-test_MiniCPM.ipynb) | First experiment: loads **MiniCPM** (`openbmb/MiniCPM5-2B`) with 🤗 `transformers`, defines a `generate_response()` helper for chat messages, and tests a system-prompt fact and a JSON-structured coding request. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonicaCoraggioso/my-first-agents/blob/main/0-test_MiniCPM.ipynb) |
| [`1-building_quasi_agent.ipynb`](1-building_quasi_agent.ipynb) | A **quasi-agent**: chains three prompts with a shared conversation memory to turn a one-line description into a function, add its documentation, write `unittest` tests, and save everything to a `.py` file. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonicaCoraggioso/my-first-agents/blob/main/1-building_quasi_agent.ipynb) |
| [`2-building_my_first_simple_agent.ipynb`](2-building_my_first_simple_agent.ipynb) | A first real **agent** that can list and read files and answer questions about them. Builds the full **agent loop** step by step: construct prompt (rules + memory) → generate response → parse `{tool_name, args}` (with error fallback) → execute the tool → update memory → continue or stop (`terminate` / max iterations), then runs it all in one `while` loop with improved agent rules (JSON tool schema + step-by-step thinking). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonicaCoraggioso/my-first-agents/blob/main/2-building_my_first_simple_agent.ipynb) |
| [`2test-physics_agent_pendulum.ipynb`](2test-physics_agent_pendulum.ipynb) | A **physics agent**: same agent loop as notebook 2, with numerical tools that simulate the nonlinear pendulum (`solve_ivp`), measure its period and compute the small-angle prediction. The agent decides which simulations to run and explains why the period grows with amplitude. Tools are verified against the exact elliptic-integral solution. Includes exercises. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonicaCoraggioso/my-first-agents/blob/main/2test-physics_agent_pendulum.ipynb) |
| [`3-building_agent_with_GAME.ipynb`](3-building_agent_with_GAME.ipynb) | A small reusable **agent framework** designed with **GAME** (Goals, Actions, Memory, Environment): `Goal`, `Action` + `ActionRegistry`, `Memory`, `Environment`, an `AgentLanguage` that builds the prompt and parses the reply, and an `Agent` class running the loop. Uses MiniCPM with prompt-based function calling (tolerant parser + retry). Example: an agent that reads the project files and writes a README. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonicaCoraggioso/my-first-agents/blob/main/3-building_agent_with_GAME.ipynb) |
| [`agent_loop_notes.md`](agent_loop_notes.md) | Full summary of the course notes for notebook 2: the six steps of the agent loop, agent rules, parsing with error fallback, execution, memory, termination, improved rules (GAIL), and lessons from running it with MiniCPM. |
| [`GAME_framework_notes.md`](GAME_framework_notes.md) | Full summary of the course notes on the GAME framework: simulating agents in a chat, the components, the agent loop, `AgentLanguage`, tool decorators and tag-based registries, plus how the notebook differs from the course. |
| `requirements.txt` | Python packages for the local environment. |
| `.gitignore` | Files Git should ignore (caches, virtual environments, `.DS_Store`, …). |

Notebooks are numbered (`0-`, `1-`, …) in the order they were written.

## Running the notebooks

### In Google Colab (easiest, free GPU)

1. Click the **Open in Colab** badge above.
2. Choose a GPU: *Runtime → Change runtime type → T4 GPU*.
3. Add a first cell with `!pip install -q transformers accelerate` (PyTorch is already installed in Colab), then run all cells.

The model weights (~5 GB) download on first run.

### Locally (Mac)

Uses conda/mamba (via mambaforge) with Python 3.11.

```bash
cd ~/my-first-agents
mamba create -n my-first-agents python=3.11 -y
conda activate my-first-agents
pip install -q -r requirements.txt
python -m ipykernel install --user --name my-first-agents --display-name "Python (my-first-agents)"
```

Check the install (prints the versions, and `True` if PyTorch can use the Apple GPU):

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.backends.mps.is_available())"
```

Next time, just run `conda activate my-first-agents`. In Jupyter or VS Code, pick the **Python (my-first-agents)** kernel.

## Notes so far

- **Chat format:** a conversation is a list of messages with a `role` (`system`, `user`, `assistant`) and `content`. `tokenizer.apply_chat_template` converts it into the format the model was trained on.
- **Answer length:** `generate_response()` defaults to `max_new_tokens=50`, which cuts answers off mid-sentence. Pass a larger value (e.g. `300`) for complete answers.
- **Structured prompts:** a Python dict can be passed to the model as text with `json.dumps`, which keeps instructions clear and machine-readable.
- **Memory = the message list:** appending each prompt *and* each answer (as an `assistant` message) to the same `messages` list lets later prompts build on earlier results (notebook 1).
- **Quasi-agent vs agent:** notebook 1 follows a fixed, hard-coded sequence of steps. A real agent also decides what to do next, uses tools and checks its own output, which matters because the model's answers vary in format (e.g. tests with or without the original function).
- **Agent loop:** construct prompt (`agent_rules + memory`) → LLM picks an action → parse → execute the matching Python function → store the result in memory → repeat until `terminate` or a maximum number of iterations (notebook 2). Action results always go back to the LLM as `user` messages.
- **Agent rules:** the system message lists the tools, the constraints and a fixed ```` ```action {...} ``` ```` JSON output format. The LLM orchestrates the tools (which one, when, with what arguments); the code executes them.
- **Better agent rules:** describing tools as a JSON schema (description + typed parameters), asking the model to think step by step before the action block, and saying explicitly when to `terminate` make the agent more reliable. The parser then has to look specifically for the ```` ```action ```` block, since the thoughts come first.
- **The LLM reasons, the code computes:** a small model is unreliable at arithmetic, so all numbers come from tools (notebook 3). Good tools return short text results, keep heavy data in code (e.g. a `sim_id` instead of a trajectory), validate their inputs, and are tested before the agent uses them.
- **GAME framework:** the Goals, Actions, Memory and Environment change from agent to agent; the loop does not. Wrapping each in a class (notebook 3) lets you build a new agent by configuring pieces instead of rewriting the loop. Swapping GPT-4o for MiniCPM only required a new `generate_response` with the same output format.
- **Extracting code:** models wrap code in Markdown fences; splitting on ```` ``` ```` and taking the first block isolates the code, but anything in later blocks is lost.
- **Hugging Face token:** without an `HF_TOKEN` the Hub shows an "unauthenticated requests" warning. It still works, but a free token gives faster downloads.

## Saving Colab work to this repository

Either use *File → Save a copy in GitHub* in Colab (then `git pull` locally), or download the `.ipynb`, move it into this folder and run:

```bash
git add .
git commit -m "Describe the change"
git push
```

Before saving, *Edit → Clear all outputs* keeps notebooks small and avoids GitHub's "Invalid Notebook" error from progress-bar widgets.
