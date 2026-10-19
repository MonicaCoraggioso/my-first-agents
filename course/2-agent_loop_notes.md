# The agent loop: notes

Summary of the course notes on building a first simple agent: the **agent loop**, agent rules, parsing, execution, memory and termination.
Companion to the notebook [`2-building_my_first_simple_agent.ipynb`](2-building_my_first_simple_agent.ipynb).

---

## 0. The big picture

The goal is an agent that can **list files**, **read them** and **answer questions** about them. The agent receives input, decides on an action, executes it and updates its memory, again and again, until the task is done.

```
┌─► 1. Construct prompt   (agent rules + memory)
│   2. Generate response  (the LLM picks the next action)
│   3. Parse response     (extract tool name + arguments)
│   4. Execute action     (run the matching Python function)
│   5. Update memory      (store the action and its result)
└── 6. Continue?          (stop on terminate or after max iterations)
```

### Quasi-agent vs agent

| | Quasi-agent (notebook 1) | Agent (notebook 2) |
|---|---|---|
| Sequence of steps | hard-coded by the programmer | decided by the LLM at each iteration |
| Tools | none | Python functions the LLM can ask to run |
| Memory | the conversation | the conversation **plus the results of actions** |
| Handling surprises | none: a fixed pipeline | adapts to new information (e.g. a missing file) |

### Glossary

| Term | Meaning |
|---|---|
| **Agent** | LLM + a loop that chooses and executes actions until the task is done |
| **Tool** | a Python function the agent can ask to run (`list_files`, `read_file`, `terminate`) |
| **Action** | the LLM's request to use a tool: `{"tool_name": ..., "args": {...}}` |
| **Environment** | what the tools act on (here: the files in the current folder) |
| **Memory** | the list of messages: task, actions taken, their results |
| **Agent rules** | the system message: role, tools, constraints, response format |

---

## 1. Step 1 · Construct the prompt

```python
prompt = agent_rules + memory
```

| Part | What it is | Why |
|---|---|---|
| `agent_rules` | a **system** message: behaviour, available tools, constraints, output format | the agent always knows *what it can do* and *how to answer* |
| `memory` | a record of all past interactions: user input, the agent's responses, results of actions | the agent knows *what just happened* and *what happened before*, which informs the next action |

- The prompt is rebuilt at **every iteration**, so the rules are never forgotten and the memory keeps growing.
- Part of the memory is the **task description**, i.e. the user's request.
- Constructing the prompt this way gives **continuity across iterations**: the agent can adapt its behaviour to previous actions and results.

### Agent rules

The rules are written as a **system message** and included at the start of every prompt. They define the agent's behaviour, capabilities and constraints, so it interacts **predictably** and within its boundaries.

````text
You are an AI agent that can perform tasks by using available tools.

Available tools:
- list_files() -> List[str]: List all files in the current directory.
- read_file(file_name: str) -> str: Read the content of a file.
- terminate(message: str): End the agent loop and print a summary to the user.

If a user asks about files, list them before reading.

Every response MUST have an action.
Respond in this format:

```action
{"tool_name": "insert tool_name", "args": {...fill in any required arguments here...}}
```
````

What the rules contain:

- **Role:** "an AI agent that can perform tasks by using available tools".
- **Tools:** a structured interface to the environment. Each tool corresponds to a **Python function** in the code.
- **Constraints:** e.g. *list files before reading them*; *every response MUST have an action*.
- **Output format:** a fixed ```` ```action {...} ``` ```` block, which makes parsing and execution easier and less error-prone.

### Key idea: reasoning vs execution

- The agent does **not** create the tools; it **orchestrates** them:
  - *which* tool to use,
  - *when* to use it,
  - *with which arguments*.
- The logic of each tool is predefined in code, so the LLM focuses on **what** to do and the code on **how**.
- **The agent adapts step by step.** Asked to read a specific file, it first lists the files, then either reads the file or reports that it doesn't exist. Every decision is based on the current state of the environment.
- **No hard-coded workflow ("glue code").** The agent works out the sequence of actions itself, effectively *building a program on top of its components*. The result is modular and extensible: more complex tasks need no rewrite of the tools.

*Example:* the user asks "What files are here?" and the expected answer is:

```action
{"tool_name": "list_files", "args": {}}
```

### Memory: example after one iteration

| # | Role | Content | Meaning |
|---|---|---|---|
| 1 | `user` | *What files are in this directory?* | the task |
| 2 | `assistant` | ```` ```action {"tool_name": "list_files", "args": {}} ``` ```` | the agent's action |
| 3 | `user` | `["file1.txt", "file2.txt"]` | the **result** of the action, fed back to the agent |

---

## 2. Step 2 · Generate the response

```python
response = generate_response(prompt)
```

- Sends the prompt (rules + memory) to the LLM.
- The LLM **decides the next action** based on the context and the rules.
- The expected output is a structured ```` ```action ```` block, which is parsed and executed in the next steps.
- **Course vs this repo:** the course calls hosted models through **LiteLLM**; here `generate_response` runs **MiniCPM** (`openbmb/MiniCPM5-2B`) locally with `transformers`. The idea is the same: messages go in, text comes out.

---

## 3. Interfacing the agent with the environment

```
LLM response ──► 3. parse ──► action {tool_name, args} ──► 4. execute ──► result ──► 5. feedback to memory
```

The LLM only produces **text**. The **interface** works out which action that text corresponds to, executes it, and then gives the agent **feedback** on the result.

---

## 4. Step 3 · Parse the response

**Goal:** extract the intended action and its parameters from the LLM's output.

- The expected format is **JSON** inside a Markdown block marked ```` ```action ````, which removes any ambiguity.
- **How:** take the content of the ```` ```action ```` block, run `json.loads`, and check that `tool_name` and `args` are both there.
- **Output:** a structured action, `{"tool_name": "list_files", "args": {}}`. `tool_name` says **which** action and `args` gives its **inputs**.

```python
def parse_action(response: str) -> Dict:
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}
```

### Fallback when the response is not valid

| Problem | Returned action |
|---|---|
| valid JSON, but no `tool_name` / `args` | `error`: *"You must respond with a JSON tool invocation."* |
| invalid JSON | `error`: *"Invalid JSON response. You must respond with a JSON tool invocation."* |

- The error goes back to the LLM **as if it came from the `user`**, asking it to retry with a valid tool call.
- This means the agent **recovers** when the LLM drifts from the format, instead of crashing.
- The course text says an invalid response "defaults to a termination action", but its code returns an **error** action. The code is what counts.

### `extract_markdown_block`

- The first version takes the **first** code block in the response.
- With the improved rules (section 9) the model writes its thoughts *before* the action, and those thoughts may contain other code blocks. The notebook's version therefore looks **specifically for** ```` ```action ```` first, and falls back to the first block.

---

## 5. Step 4 · Execute the action

**Goal:** run the Python function that matches `tool_name`, with the parsed `args`.

```python
if action["tool_name"] == "list_files":
    result = {"result": list_files()}
elif action["tool_name"] == "read_file":
    result = {"result": read_file(action["args"]["file_name"])}
elif action["tool_name"] == "error":
    result = {"error": action["args"]["message"]}
elif action["tool_name"] == "terminate":
    print(action["args"]["message"])
    break
else:
    result = {"error": "Unknown action: " + action["tool_name"]}
```

| `tool_name` | What happens | Result |
|---|---|---|
| `list_files` | call `list_files()` | `{"result": [files...]}` |
| `read_file` | call `read_file(args["file_name"])` | `{"result": file content}` |
| `error` | comes from the parser | `{"error": message}` |
| `terminate` | print the final message | exit the loop (`break`) |
| anything else | unknown tool | `{"error": "Unknown action: ..."}` |

- This is where the agent does **tangible work**: interacting with files, printing messages.
- It **connects the decision** made by the LLM **with concrete results**, which go back into memory for the next iteration.
- The tools are ordinary functions: `list_files()` uses `os.listdir(".")`, and `read_file()` opens the file and returns an error message if it's missing.

---

## 6. Step 5 · Update the memory

```python
memory.extend([
    {"role": "assistant", "content": response},
    {"role": "user", "content": json.dumps(result)}
])
```

| Role | Content | Meaning |
|---|---|---|
| `assistant` | the LLM's `response` | the agent's **intention**, i.e. the action it chose |
| `user` | `json.dumps(result)` | the **feedback**: what happened when the action ran |

- Memory is the agent's record of the interaction: user requests, actions performed, and their outcomes.
- Results of actions are **always** sent back with the **`user`** role, so the LLM clearly understands what happened after each action.
- The growing history gives **continuity**: the agent refines its behaviour as memory grows and keeps track of the status of its work.

---

## 7. Step 6 · Decide whether to continue

```python
if action["tool_name"] == "terminate":
    print(action["args"]["message"])
    break
```

- **Stop** if:
  - the action is `terminate`: the loop exits and the closing message from the action's arguments is printed;
  - a predefined limit is reached, e.g. a **maximum number of iterations** (a safety net against endless loops).
- **Otherwise** go back to step 1 with the updated memory, for the next user request or the next part of the task.

### Example: iterative adaptation

The task is *read a file*, but no file name was given.

1. **Iteration 1:** `list_files` returns the available files.
2. **Iteration 2:** based on that result in memory, the agent asks the user to pick a file.
3. This continues until the task is done or no further action is needed.

- **Memory is what decides when to stop.** At each step the agent looks back at it to check whether the overall task is complete.
- It's a balance: adapt to new information, but eventually conclude.
- Stopping rules can also be **written into the agent rules**, e.g. *stop after more than two errors*.
- ⚠️ With only `list_files`, `read_file` and `terminate`, the agent **cannot actually ask the user** anything. It would have to `terminate` with a question. A real "ask the user" step needs an `ask_user` tool.

---

## 8. The full loop

```python
max_iterations = 10
iterations = 0
user_task = input("What would you like me to do? ")
memory = [{"role": "user", "content": user_task}]       # memory starts with only the task

while iterations < max_iterations:
    prompt = agent_rules + memory                          # 1. construct prompt
    response = generate_response(prompt)                   # 2. generate response
    action = parse_action(response)                        # 3. parse
    if action["tool_name"] == "list_files":                # 4. execute
        result = {"result": list_files()}
    elif action["tool_name"] == "read_file":
        result = {"result": read_file(action["args"]["file_name"])}
    elif action["tool_name"] == "error":
        result = {"error": action["args"]["message"]}
    elif action["tool_name"] == "terminate":               # 6. stop
        print(action["args"]["message"])
        break
    else:
        result = {"error": "Unknown action: " + action["tool_name"]}
    memory.extend([                                        # 5. update memory
        {"role": "assistant", "content": response},
        {"role": "user", "content": json.dumps(result)},
    ])
    iterations += 1                                        # 6. max iterations
```

The course version also sets `result = "Action executed"` and checks `terminate` a second time after updating memory. Both are redundant: `result` is always overwritten, and `terminate` has already exited the loop with `break`.

---

## 9. Improved agent rules

| | First rules | Improved rules |
|---|---|---|
| **Tools** | one line each: `list_files() -> List[str]: ...` | a **JSON schema**: `description` + `parameters` (name, `type`, `description`) |
| **Reading files** | *list them before reading* | *if asked about files, documents or content → list first, then read* |
| **Ending** | `terminate` only listed as a tool | explicit: *when done, use `terminate`* |
| **Response format** | action block only | **think step by step first**, then the action block |
| **Emphasis** | *Every response MUST have an action* | *Important!!! … You must ALWAYS respond in this format* |

````text
<Stop and think step by step. Parameters map to args. Insert a rich description of your step by step thoughts here.>

```action
{"tool_name": "insert tool_name", "args": {...fill in any required arguments here...}}
```
````

Why these changes help:

- **JSON tool schema:** the same structured format real LLM APIs use for function calling. Parameter names and types are unambiguous ("parameters map to args").
- **Thinking step by step** before acting (chain of thought) leads to a better choice of tool and arguments.
- **Explicit termination:** the agent knows how and when to finish, instead of running until `max_iterations`.

### Prompt structure: GAIL

| Letter | Part of the prompt | In the improved rules |
|---|---|---|
| **G**oals | role and task | "You are an AI agent that can perform tasks…" + the user's request |
| **A**ctions | the tool list | the JSON schema of `list_files`, `read_file`, `terminate` |
| **I**nformation | context | the memory (task + actions + results) |
| **L**anguage | response format | thoughts + the ```` ```action ```` block |

---

## 10. Lessons from running it (with MiniCPM)

These come from an early physics version of this agent (a nonlinear-pendulum agent using the same loop), which led to the project in [`../projects/`](../projects/):

- **The reasoning can be right while the format is wrong.** MiniCPM planned the steps correctly but wrapped its action in `<action>…</action>` tags instead of ```` ```action ````. The parser rejected every answer.
- **The angle-bracket template line** (`<Stop and think…>`) probably invited the XML-style tags.
- **Error messages must be actionable.** "Invalid JSON response" doesn't say what was wrong, so the model repeated the same mistake every iteration.
- **Fixes:**
  - a **tolerant parser** that also accepts `<action>` tags;
  - an error message that **shows the correct format**;
  - a template **without angle brackets**.
- **Always print the raw responses** (`Agent response: ...`). They are what made the problem visible.
- **Tools should be robust:** convert the argument types, validate ranges, and return errors as text rather than crashing. A `try/except` around the tool call turns a bad argument into feedback the agent can act on.
- **Results go into memory,** so keep them short. Return a summary or an id rather than raw data.

---

## 11. Known limitations of the simple agent

| Limitation | Consequence | Fix (often in the GAME framework, notebook 3) |
|---|---|---|
| tools listed by hand in **three places** (rules, function, `if/elif`) | easy to get them out of sync | `Action` objects + `ActionRegistry`; tool decorators |
| `if/elif` chain inside the loop | the loop changes every time a tool is added | an `Environment` that runs any action |
| missing argument → `KeyError` | the loop crashes | `try/except` around execution |
| `read_file` can open `../...` and large files | safety and context-size issues | restrict paths, truncate content |
| no `ask_user` tool | the agent can't really ask questions | add the tool |
| the `user` role is used for results | works, but mixes user input with tool output | real APIs have a dedicated *tool* role |


