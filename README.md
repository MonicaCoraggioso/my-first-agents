# my-first-agents

A sandbox for testing code.

## Environment setup

Uses conda/mamba (already installed via mambaforge) with Python 3.11.

```bash
cd ~/my-first-agents
mamba create -n my-first-agents python=3.11 -y
conda activate my-first-agents
pip install -q -r requirements.txt
python -m ipykernel install --user --name my-first-agents --display-name "Python (my-first-agents)"
python check_env.py
```

Next time, just run `conda activate my-first-agents`.

In Jupyter or VS Code, pick the **Python (my-first-agents)** kernel.
