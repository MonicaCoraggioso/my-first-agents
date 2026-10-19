# GAME framework: notes

Summary of the course notes on designing and implementing agents with **GAME** (Goals, Actions, Memory, Environment).
Companion to the notebook [`3-building_agent_with_GAME.ipynb`](3-building_agent_with_GAME.ipynb).

---

## 0. The big picture

| | Component | Answers | In code |
|---|---|---|---|
| **G** | Goals | *What* should the agent achieve, and *how*? | `Goal` |
| **A** | Actions | What *can* it do? | `Action`, `ActionRegistry` |
| **M** | Memory | What has happened so far? | `Memory` |
| **E** | Environment | How are actions actually executed? | `Environment` |

Two more pieces connect them:

- **`AgentLanguage`**: translates the GAME objects into a prompt, and the LLM's reply back into an action.
- **`Agent`**: runs the loop (construct prompt → generate → parse → execute → update memory → terminate?).

**Core idea:** the GAME pieces are what *change* from one agent to another, while the loop stays *the same*. So the code should mirror the design. Write the loop once and make every GAME piece swappable.

**Workflow suggested by the notes:**

1. Design the GAME pieces.
2. **Simulate** the agent in a chat to test the design.
3. Implement it with the framework.
4. Refine the tools, goals and language.

---

## 1. Simulate before you code

Test a GAME design in a chat interface (ChatGPT, Claude, …) **before** writing any code. Problems are cheapest to fix at this stage.

### What a simulation checks

1. The goals are **achievable** with the planned actions.
2. The **memory** requirements are reasonable.
3. The actions are **sufficient** to solve the problem.
4. The agent can make **good decisions** with the information it gets.

### Template prompt

```
I'd like to simulate an AI agent that I'm designing. The agent will be built using these components:

Goals: [list your goals]
Actions: [list available actions]

At each step, your output must be an action to take.
Stop and wait and I will type in the result of the action as my next message.

Ask me for the first task to perform.
```

The LLM plays the agent, and **you play the environment**: you type in the result of each action. Example goals for a "Proactive Coder" agent are *find potential code enhancements*, *keep changes small and self-contained*, *get user approval before changing anything* and *keep existing interfaces*. Its actions would be `list_project_files()`, `read_project_file(filename)`, `ask_user_approval(proposal)` and `edit_project_file(filename, changes)`.

### What you learn

- **How the agent reasons.** Does it list the files first, or jump straight to reading? This shows whether the goals and actions lead to systematic problem solving.
- **How much context results need.** Compare returning `["main.py", "utils.py"]` with returning `{"files": [...], "total_files": 3, "directory": "/project"}`. Extra metadata often leads to better decisions.
- **Tool descriptions get better.** `read_project_file(filename) -> returns the file content` might become *"Returns the content of a Python file from the project directory. The filename should be one previously returned by list_project_files()."*
- **Goals get more specific.** *"Find potential code enhancements"* might become *"Identify opportunities to improve error handling and input validation in functions."*
- **Memory = the chat.** Each exchange is one loop iteration plus one memory entry, so you can see how much history the agent can handle.
- **Learning from failures.** Return errors (`{"error": "FileNotFoundError: ..."}`) or broken data (`{"cont3nt": ...}`). Does the agent try another approach, give up, or stay focused on its goal?
- **Preventing runaway agents.** Try termination rules safely, e.g. stop after N files or N suggestions.
- **Fast iteration.** You can test "a project with 100 files" just by claiming that's what the tool returned.
- **Ask the agent afterwards.** Which tools did it miss? What was unclear? Which goals were vague? Its suggestions are often good, for example *"ask_user_approval should include code snippets"*.
- **Build an example library.** Save the good and bad exchanges. They become prompt examples and test cases later.
  - *Good:* reads `utils.py` → notices missing input validation → proposes focused improvements.
  - *Poor:* "I'll start editing all the files", skipping analysis and user approval.

> Applied to this repo: the pendulum notebook's failed run (`<action>` tags instead of ```` ```action ````) is exactly the kind of problem a quick chat simulation would have revealed.

---

## 2. The components (framework, part 1)

At first the framework *adds* complexity. This buys flexibility and reuse, so new agents are built by **configuring** pieces rather than rewriting the loop.

### G · Goals

```python
@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str
```

- A goal covers both **what** to achieve and **how** to approach it.
- Objects instead of one big "wall of text" of instructions.
- `priority` decides which goal comes first and how goals are sorted or formatted in the prompt.
- Kinds of goals:
  - the task itself;
  - **examples** of how to reason in certain situations;
  - **core rules** shared by all agents in a system;
  - special instructions for particular kinds of task.

```python
file_management_goal = Goal(
    priority=1,
    name="file_management",
    description="""Manage files in the current directory by:
    1. Listing files when needed
    2. Reading file contents when needed
    3. Searching within files when information is required
    4. Providing helpful explanations about file contents""")
```

### A · Actions

- Actions are the agent's **toolkit**: discrete capabilities that run in the environment. They form the **interface** between the agent and the world.
- **`Action`** bundles `name`, `function`, `description`, `parameters` (a JSON schema) and `terminal` (whether running it ends the loop). `execute(**args)` calls the function.
- **`ActionRegistry`**: the LLM replies with a *name* in JSON, and the registry finds the matching `Action` object. Its methods are `register`, `get_action(name)` (returns `None` if the name is unknown) and `get_actions()`.
- **Parameters** use the standard JSON schema format:
  ```python
  {"type": "object",
   "properties": {"file_name": {"type": "string", "description": "Name of the file to read"}},
   "required": ["file_name"]}
  ```
- Example file tools: `list_files()`, `read_file(file_name)` and `search_in_file(file_name, search_term)`, which returns `(line number, line)` pairs.

### M · Memory

```python
class Memory:
    def __init__(self): self.items = []
    def add_memory(self, memory: dict): self.items.append(memory)
    def get_memories(self, limit=None): return self.items[:limit]
```

- **Why wrap a plain list?** So features can be added later **without changing the loop**: storing memory in a database, choosing which memories the agent sees at each iteration, or different strategies as subclasses.
- **The prompt always needs a list of messages,** however the memory is stored (even as a graph). `get_memories()` is the single access point, which keeps `Memory` independent of how it's used.
- ⚠️ The notes say it returns the **last N** messages, but `items[:limit]` returns the **first** N. The fix is `items[-limit:]`.

### E · Environment

```python
class Environment:
    def execute_action(self, action, args) -> dict:
        try:    return self.format_result(action.execute(**args))
        except Exception as e:
            return {"tool_executed": False, "error": str(e), "traceback": traceback.format_exc()}
    def format_result(self, result) -> dict:
        return {"tool_executed": True, "result": result, "timestamp": time.strftime(...)}
```

- It replaces the hard-coded `if/elif` chain. The loop runs actions **without knowing how they work** and without any conditional logic.
- It's the **bridge** between the agent and the outside world.
- Errors are **returned as data**, not raised, so the agent can see what went wrong and adapt.

---

## 3. The agent and its loop (framework, part 2)

`Agent(goals, agent_language, action_registry, generate_response, environment)` holds the GAME pieces and offers a simple interface: `agent.run(user_input, memory=None, max_iterations=50)`, which returns a `Memory`. The memory is passed to `run()` rather than stored in the agent.

### The six steps

| Step | Method | What happens |
|---|---|---|
| (start) | `set_current_task` | the task goes into memory as a `user` item |
| 1. Construct prompt | `construct_prompt` | the `AgentLanguage` turns goals, actions, memory (and environment details) into a `Prompt` |
| 2. Generate response | `prompt_llm_for_action` | calls `generate_response`, a plain function passed in at creation, so **any LLM provider** works |
| 3. Parse response | `get_action` | `parse_response` gives the **invocation** (`{"tool", "args"}`); the registry gives the **action** |
| 4. Execute | `environment.execute_action` | runs it; this could mean API calls, file input/output, database queries or data processing |
| 5. Update memory | `update_memory` | stores the decision (`assistant`) and the result, which become context for the next prompt |
| 6. Terminate? | `should_terminate` | stops if the chosen action is `terminal`, or after `max_iterations` |

- **Action vs invocation:** the *action* is the interface definition of what the agent **can** do; the *invocation* is the specific arguments it **chose** this time.
- **Possible improvement:** at step 3, validate the invocation's arguments against the action's JSON schema.
- **Actions** are *defined* in the registry but *executed* within the environment, which provides the resources.

### Flow of information in one iteration

1. **Memory** gives the user's request plus past decisions and results.
2. **Goals** say what to achieve and the rules for how.
3. The **ActionRegistry** says what the agent can do, and looks up actions by name.
4. The **AgentLanguage** formats memory, actions and goals into a prompt.
5. The **LLM** chooses an action.
6. The **AgentLanguage** parses the response into an invocation (usually from a tool call).
7. The **Environment** executes the action.
8. The **result** goes back into memory.
9. The loop **repeats** until a terminal tool is called or `max_iterations` is reached.

### Specialized agents

Same loop, different GAME pieces:

```python
research_agent = Agent(goals=[...find and summarize info on X...], agent_language=ResearchLanguage(),
                       action_registry=..., generate_response=openai_call, environment=WebEnvironment())
coding_agent   = Agent(goals=[...write and debug code for Y...],    agent_language=CodingLanguage(),
                       action_registry=..., generate_response=anthropic_call, environment=DevEnvironment())
```

(These are sketches; those classes don't exist in the notebook.)

---

## 4. AgentLanguage: the translator

It sits between the structured GAME world and the LLM's text-based world, like an interpreter. There are many ways to ask an LLM for its next action (plain text you parse, or function calling) and many ways to show it memory (one long string, or separate ChatML messages). The `AgentLanguage` turns these choices into **reusable strategies** that plug into the agent.

### Two responsibilities

1. **Prompt construction:** turn the GAME components into something the LLM understands.
2. **Response parsing:** interpret the LLM's reply as an action.

```python
class AgentLanguage:
    def construct_prompt(self, actions, environment, goals, memory) -> Prompt: raise NotImplementedError
    def parse_response(self, response: str) -> dict: raise NotImplementedError
```

These are the two points in the loop where the language steps in: step 1 (`construct_prompt`) and step 3 (`get_action` → `parse_response`).

- **Construction** = `format_goals` (instructions) + `format_actions` (tool descriptions) + `format_memory` (conversation context). With function calling, the result is `{"messages": [...], "tools": [...]}`.
- **Parsing:** for example, extract the JSON from a chatty reply that contains a ```` ```action ```` block. Whatever style the LLM uses (function calling, JSON blocks, natural language), **the loop doesn't change**.

### Two example languages

| | `AgentJsonActionLanguage` | `AgentFunctionCallingActionLanguage` |
|---|---|---|
| Actions shown as | system message: `Available Tools: [...]` + the required format | `tools=[{"type": "function", "function": {...}}]` |
| LLM replies with | thoughts + ```` ```action {"tool", "args"} ``` ```` | a structured tool call |
| Parsing | find the ```` ```action ```` block → `json.loads` (raises on failure) | `json.loads`; if it fails → `terminate` with the text as the message |
| Pro | **the reasoning is visible** | **valid JSON** is easy to get |
| Con | free text is fragile to parse | the reasoning is not visible |

The JSON action format template:

````
<Stop and think step by step. Insert your thoughts here.>

```action
{"tool": "tool_name", "args": {...fill in arguments...}}
```
````

### Why swappable languages are useful

- **The same agent behaves differently** depending only on its language, e.g. `AgentJsonActionLanguage` for simple tasks and function calling for complex ones.
- **You can:**
  - try different prompt formats without changing the agent's logic;
  - support LLM providers with different styles and play to each one's strengths;
  - add new response formats without touching existing code;
  - **handle errors and retries at the language level.**
- **In short:** all communication logic lives in one place, it's easy to experiment, structured formats make the agent more reliable, and the approach can evolve as LLMs improve. "How to communicate" is kept separate from "what to do".

---

## 5. Tool decorators: keeping code and documentation in sync

**The problem:** when an action is registered by hand, the schema and description live apart from the function. Add a parameter or change the behaviour, and they **go out of sync**, so the agent calls tools incorrectly. These bugs are hard to find.

**The solution:** a decorator builds the action from the function itself:

```python
@register_tool(tags=["file_operations"])
def read_file(file_path: str) -> str:
    """Reads and returns the content of a file from the specified path.

    Args:
        file_path: The path to the file to read
    Returns:
        The contents of the file as a string
    """
    with open(file_path) as f:
        return f.read()
```

**What `register_tool` does automatically:**

1. uses the **function name** as the tool name;
2. uses the **docstring** as the description;
3. builds the **JSON schema** from the type hints and parameters;
4. **registers** the tool in a global `tools` dict, and also in `tools_by_tag` for each tag;
5. returns the function **unchanged**, so it can still be called normally.

It accepts optional overrides: `tool_name`, `description`, `parameters_override`, `terminal` and `tags`.

**How `get_tool_metadata` builds the schema:**

- `inspect.signature(func)` lists the parameters, and `get_type_hints(func)` gives their types.
- Each Python type becomes a JSON type (`get_json_type`), and the result is `{"type": "object", "properties": {...}, "required": [...]}`.
- A parameter with **no default value is required**.
- Special context parameters (`action_context`, `action_agent`) are skipped.

**Why decorators are common in agent frameworks:**

1. **Single source of truth:** the docstring says what the tool does, the type hints define its parameters, and the code shows how it works.
2. **Automatic updates:** add `encoding: str = 'utf-8'` and the schema gains an *optional* `encoding` parameter, with no registration code to edit.
3. **Better organization:** tags group related tools.
4. **Better development experience:** you just write normal, documented Python functions.

---

## 6. Organizing tools: from tags to registries

It's like a workshop where tools are sorted into drawers by purpose. There are three layers:

1. **Tool decorators:** tag and document each tool.
2. **Tool registry:** central storage of *all* tools, in `tools` (by name) and `tools_by_tag` (by tag).
3. **Action registry:** a *curated* subset for one agent.

```python
@register_tool(tags=["file_operations"])          def read_file(file_path: str) -> str: ...
@register_tool(tags=["file_operations", "write"]) def write_file(file_path: str, content: str) -> None: ...
@register_tool(tags=["database", "read"])         def query_database(query: str) -> List[Dict]: ...

# tools_by_tag == {"file_operations": ["read_file", "write_file"], "write": ["write_file"],
#                  "database": ["query_database"], "read": ["query_database"]}
```

**Focused agents** are built by choosing tags:

```python
file_agent      = Agent(goals=[Goal(1, "File Handler", "Manage file operations")],
                        agent_language=AgentFunctionCallingActionLanguage(),
                        action_registry=ActionRegistry(tags=["file_operations"]),
                        generate_response=generate_response, environment=Environment())

read_only_agent = Agent(goals=[Goal(1, "Read Only", "Read but don't modify data")],
                        ..., action_registry=ActionRegistry(tags=["read"]), ...)
```

The second example also gives **safety by construction**: a read-only agent simply never receives write tools.

> The `ActionRegistry` in the notebook has no `tags` option yet. Adding it, together with `register_tool`, is a natural next step.

---

## 7. Key benefits of the GAME framework

1. **Better organization:** each component has a clear, separate purpose.
2. **Reusability:** you can swap components (actions, environment, LLM, language) without changing the core logic.
3. **Extensibility:** new goals and actions are easy to add.
4. **Standard interface:** every agent is used the same way, `agent.run(task)` → `Memory`.
5. **Memory management:** the framework updates memory automatically.

The result is easier to understand and maintain, especially as an agent grows. The cost is more code and more layers at the start, which pays off from the second agent onward.

**Checklist for a new agent:**

1. Simulate it in a chat first.
2. Write clear goals, including when to stop.
3. Write tools with precise descriptions and accurate schemas (or use decorators).
4. Make the results the right size: enough context, but not too much.
5. Return errors as data.
6. Have a terminal action plus `max_iterations`.
7. Read the raw LLM responses when something goes wrong.
