"""A small GAME agent framework (Goals, Actions, Memory, Environment) for local LLMs.

Adapted from the Coursera course "AI Agents and Agentic AI with Python & Generative AI"
(Vanderbilt University), with changes for a small local model (MiniCPM):

- `generate_response` emulates function calling in the prompt (MiniCPM has no
  function-calling API): tool schemas go into the system message, a tolerant parser
  extracts the tool call, and the model gets one retry with an actionable error message.
- Environment results are sent back to the LLM as `user` messages.
- `SafeAgent` never crashes on an unknown tool name or a malformed invocation.

Usage:

    from agent_framework import (load_minicpm, generate_response, Goal, Action,
                                 ActionRegistry, Environment, SafeAgent,
                                 AgentFunctionCallingActionLanguage)

    load_minicpm()                                   # once, before running an agent
    registry = ActionRegistry()
    registry.register(Action(name=..., function=..., description=..., parameters=...))
    agent = SafeAgent(goals, AgentFunctionCallingActionLanguage(), registry,
                      generate_response, Environment())
    memory = agent.run("the task", max_iterations=30)
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# ----------------------------------------------------------------------------
# The LLM (MiniCPM, local)
# ----------------------------------------------------------------------------

_tokenizer = None
_model = None


def load_minicpm(model_id: str = "openbmb/MiniCPM5-2B"):
    """Load the model and tokenizer once (~5 GB, downloaded on first run, then cached).

    Returns (tokenizer, model). They are also stored in this module, so that
    `minicpm_chat` and `generate_response` can use them.
    """
    global _tokenizer, _model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",   # keep the precision the weights were saved in
        device_map="auto",    # use the GPU automatically when available
    )
    print("Model loaded successfully!")
    return _tokenizer, _model


def minicpm_chat(messages: List[Dict], max_new_tokens: int = 1024) -> str:
    """Send chat messages to MiniCPM and return its reply as text."""
    if _model is None:
        raise RuntimeError("Call load_minicpm() first.")
    import torch

    # Chat messages -> the model's chat format -> token IDs, on the model's device
    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,   # add the "assistant's turn" marker
        enable_thinking=False,        # skip MiniCPM's hidden reasoning step
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():             # no gradients: we are not training
        outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the new tokens (drop the prompt)
    return _tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ----------------------------------------------------------------------------
# Prompt and emulated function calling
# ----------------------------------------------------------------------------

@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)   # goals + memory, as chat messages
    tools: List[Dict] = field(default_factory=list)      # actions, as JSON schemas
    metadata: dict = field(default_factory=dict)         # default_factory avoids a shared mutable default


def tools_to_instructions(tools: List[Dict]) -> str:
    """Describe the tools in plain text, since MiniCPM has no function-calling API."""
    tool_list = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "parameters": t["function"]["parameters"],
        }
        for t in tools
    ]
    return (
        "You can use the following tools:\n"
        "```json\n" + json.dumps(tool_list, indent=2) + "\n```\n\n"
        "Every response MUST call exactly one tool. First think step by step, "
        "then write the tool call as a Markdown code block in exactly this format "
        "(a code block, not XML tags):\n"
        "```action\n"
        '{"tool": "<tool name>", "args": {<arguments>}}\n'
        "```"
    )


def extract_tool_call(text: str, tool_names: List[str]):
    """Find a tool call in the model's text. Returns {"tool": ..., "args": ...} or None.

    Tolerant: tries ```action, <action>...</action>, ```json and finally the whole text.
    """
    candidates = []
    if "```action" in text:
        candidates.append(text.split("```action", 1)[1].split("```", 1)[0])
    if "<action>" in text:
        candidates.append(text.split("<action>", 1)[1].split("</action>", 1)[0])
    if "```json" in text:
        candidates.append(text.split("```json", 1)[1].split("```", 1)[0])
    candidates.append(text)

    for candidate in candidates:
        try:
            data = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            name = data.get("tool") or data.get("tool_name") or data.get("name")
            if name in tool_names:
                return {"tool": name, "args": data.get("args") or data.get("arguments") or {}}
    return None


def generate_response(prompt: Prompt, max_retries: int = 1) -> str:
    """Call the LLM. Returns a tool call as a JSON string, or plain text.

        '{"tool": "<name>", "args": {...}}'  if the model called a tool,
        the model's text                      otherwise.
    """
    messages = [dict(m) for m in prompt.messages]          # copy: don't modify the prompt

    # No tools: just chat
    if not prompt.tools:
        return minicpm_chat(messages)

    # 1. Tools -> text instructions, appended to the system message
    instructions = tools_to_instructions(prompt.tools)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + instructions
    else:
        messages.insert(0, {"role": "system", "content": instructions})

    tool_names = [t["function"]["name"] for t in prompt.tools]

    for attempt in range(max_retries + 1):
        # 2. Generate
        text = minicpm_chat(messages)

        # 3. Extract the tool call
        call = extract_tool_call(text, tool_names)
        if call is not None:
            return json.dumps(call)

        # 4. Retry with an actionable error message
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "Your answer did not contain a valid tool call. Available tools: "
                + ", ".join(tool_names) + ". Reply with one tool call in this format:\n"
                '```action\n{"tool": "<tool name>", "args": {...}}\n```'},
        ]

    return text   # still no tool call: return the text (the agent language will handle it)


# ----------------------------------------------------------------------------
# G · A · M · E
# ----------------------------------------------------------------------------

@dataclass(frozen=True)          # frozen: a goal is immutable once created
class Goal:
    priority: int                # order of importance (used to sort/format goals)
    name: str                    # short label
    description: str             # what to achieve and how


class Action:
    """One tool the agent can use: its Python function + the description the LLM sees."""

    def __init__(self,
                 name: str,               # name the LLM uses to call it
                 function: Callable,      # Python function that does the work
                 description: str,        # what it does (shown to the LLM)
                 parameters: Dict,        # JSON schema of the arguments (shown to the LLM)
                 terminal: bool = False): # True -> running it ends the agent loop
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)


class ActionRegistry:
    """Lookup table name -> Action."""

    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> [Action, None]:
        return self.actions.get(name, None)          # None if the LLM invents a tool name

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())


class Memory:
    """The agent's history: a list of dicts like {"type": "assistant", "content": "..."}."""

    def __init__(self):
        self.items = []  # basic conversation history

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        # Note: items[:limit] returns the FIRST `limit` items. To get the most
        # recent ones, it would be items[-limit:]. With limit=None both return everything.
        return self.items[:limit]

    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory


class Environment:
    """Runs actions and wraps their results (or errors) in a standard dict."""

    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            # The error goes back to the agent instead of crashing the loop
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }


# ----------------------------------------------------------------------------
# Agent language: GAME <-> prompt
# ----------------------------------------------------------------------------

class AgentLanguage:
    """Interface: how GAME objects become a prompt, and how the reply becomes an action."""

    def construct_prompt(self, actions: List[Action], environment: Environment,
                         goals: List[Goal], memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


class AgentFunctionCallingActionLanguage(AgentLanguage):
    """Language for function calling: goals -> system message, actions -> tool schemas."""

    def format_goals(self, goals: List[Goal]) -> List:
        # All goals -> one system message: "name:\n----\ndescription\n----"
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join([f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        return [{"role": "system", "content": goal_instructions}]

    def format_memory(self, memory: Memory) -> List:
        """Map memory items to chat messages."""
        # assistant decisions -> role "assistant"; environment results and user messages -> role "user"
        mapped_items = []
        for item in memory.get_memories():
            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)
            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})
        return mapped_items

    def format_actions(self, actions: List[Action]) -> List:
        """Actions -> tool schemas in the standard function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024],   # up to 1024 characters
                    "parameters": action.parameters,
                },
            }
            for action in actions
        ]

    def construct_prompt(self, actions: List[Action], environment: Environment,
                         goals: List[Goal], memory: Memory) -> Prompt:
        # messages = goals (system) + memory (conversation); tools = actions
        prompt = self.format_goals(goals) + self.format_memory(memory)
        return Prompt(messages=prompt, tools=self.format_actions(actions))

    def parse_response(self, response: str) -> dict:
        """Parse the LLM response: a JSON tool call, or plain text -> terminate."""
        try:
            return json.loads(response)
        except Exception:
            # Not a tool call: treat the text as the final answer
            return {"tool": "terminate", "args": {"message": response}}


# ----------------------------------------------------------------------------
# The agent: the loop
# ----------------------------------------------------------------------------

class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment):
        """Initialize an agent with its core GAME components."""
        self.goals = goals                          # G
        self.generate_response = generate_response  # the LLM
        self.agent_language = agent_language        # GAME <-> prompt translator
        self.actions = action_registry              # A
        self.environment = environment              # E   (M is passed to run())

    def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(), environment=self.environment, goals=goals, memory=memory)

    def get_action(self, response):
        # Reply -> {"tool": ..., "args": ...} -> the registered Action object
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """Update memory with the agent's decision and the environment's response."""
        memory.add_memory({"type": "assistant", "content": response})
        memory.add_memory({"type": "environment", "content": json.dumps(result)})

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        return self.generate_response(full_prompt)

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """Execute the GAME loop for this agent with a maximum iteration limit."""
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for _ in range(max_iterations):
            # 1. Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(self.goals, memory, self.actions)

            # 2. Generate a response from the agent
            print("Agent thinking...")
            response = self.prompt_llm_for_action(prompt)
            print(f"Agent Decision: {response}")

            # 3. Determine which action the agent wants to execute
            action, invocation = self.get_action(response)

            # 4. Execute the action in the environment
            result = self.environment.execute_action(action, invocation["args"])
            print(f"Action Result: {result}")

            # 5. Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # 6. Check if the agent has decided to terminate
            if self.should_terminate(response):
                break

        return memory


class SafeAgent(Agent):
    """Agent that never crashes on an unknown tool name or a malformed invocation."""

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)

        # Malformed invocation (e.g. valid JSON without "tool") -> treat the text as the final answer
        if not isinstance(invocation, dict) or "tool" not in invocation:
            invocation = {"tool": "terminate", "args": {"message": str(response)}}
        invocation.setdefault("args", {})

        action = self.actions.get_action(invocation["tool"])
        if action is None:
            # Unknown tool -> an action whose result tells the agent what went wrong
            name = invocation["tool"]
            available = [a.name for a in self.actions.get_actions()]
            action = Action(
                name="unknown_tool",
                function=lambda **_: f"Error: unknown tool '{name}'. Available tools: {available}",
                description="", parameters={}, terminal=False,
            )
        return action, invocation
