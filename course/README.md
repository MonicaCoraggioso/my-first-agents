# course/

Notebooks and notes written while following this course:

> **AI Agents and Agentic AI with Python & Generative AI**
> Dr. Jules White, Vanderbilt University, on Coursera
> https://www.coursera.org/learn/ai-agents-python
>
> First course of the specialization **AI Agents and Agentic AI in Python: Powered by Generative AI**
> https://www.coursera.org/specializations/ai-agents-python

The ideas, structure and much of the code come from the course. 

## Contents

| # | Notebook | Notes | Topic |
|---|---|---|---|
| 0 | [`0-test_MiniCPM.ipynb`](0-test_MiniCPM.ipynb) | | running a local LLM, chat messages and roles |
| 1 | [`1-building_quasi_agent.ipynb`](1-building_quasi_agent.ipynb) | | chaining prompts with a shared memory (fixed sequence of steps) |
| 2 | [`2-building_my_first_simple_agent.ipynb`](2-building_my_first_simple_agent.ipynb) | [`2-agent_loop_notes.md`](2-agent_loop_notes.md) | the agent loop: prompt → response → parse → execute → memory → stop |
| 3 | [`3-building_agent_with_GAME.ipynb`](3-building_agent_with_GAME.ipynb) | [`3-GAME_framework_notes.md`](3-GAME_framework_notes.md) | a reusable framework: Goals, Actions, Memory, Environment |

## How my version differs from the course

- **LLM:** the course calls hosted models (e.g. GPT-4o) through LiteLLM. Here every notebook uses **MiniCPM** (`openbmb/MiniCPM5-2B`), an open-source model running locally with `transformers`. No API key is needed.
- **Notes:** each `*_notes.md` file summarises the course material for that notebook, together with what I learned running it with a small model.

My own project built on top of this material is in [`../projects/`](../projects/).
