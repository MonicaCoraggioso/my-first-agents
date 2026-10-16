# my-first-agents

A sandbox for learning to build AI agents with small open-source language models, starting with running a chat model locally or in Google Colab.

## Contents

| File | What it is |
|---|---|
| [`0-test_MiniCPM.ipynb`](0-test_MiniCPM.ipynb) | First experiment: loads **MiniCPM** (`openbmb/MiniCPM5-2B`) with 🤗 `transformers`, defines a `generate_response()` helper for chat messages, and tests a system-prompt fact and a JSON-structured coding request. 
| [`1-building_quasi_agent.ipynb`](1-building_quasi_agent.ipynb) | A **quasi-agent**: chains three prompts with a shared conversation memory to turn a one-line description into a function, add its documentation, write `unittest` tests, and save everything to a `.py` file. 
| [`2-building_my_first_simple_agent.ipynb`](2-building_my_first_simple_agent.ipynb) | A first real **agent** that can list and read files and answer questions about them. Builds the full **agent loop** step by step: construct prompt (rules + memory) → generate response → parse `{tool_name, args}` (with error fallback) → execute the tool → update memory → continue or stop (`terminate` / max iterations), then runs it all in one `while` loop. 
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


Before saving, *Edit → Clear all outputs* keeps notebooks small and avoids GitHub's "Invalid Notebook" error from progress-bar widgets.
