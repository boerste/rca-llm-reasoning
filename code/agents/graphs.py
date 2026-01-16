import operator
import logging
import time
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolCall
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from typing_extensions import Any, Annotated, TypedDict, List, Tuple, Literal, Callable

from agents.structured_output import StructuredResponse
from utils.common import retrieve_non_think_content
from utils.llm import get_message_content_as_string

annoying_pydantic = "For further information visit https://errors.pydantic.dev/2.11/v/missing"

# ---------------------------------------- INTERNAL STATES ----------------------------------------    

# Inherit 'messages' key from MessagesState
class CustomMessagesState(MessagesState):
    """
    Messages state extended with tool retry and structured outputs.

    Includes `failed_tool_calls` for retry flows and `final_response` to
    capture structured outputs independent of the chat history.
    """
    # For tool retry
    failed_tool_calls: Annotated[list[ToolCall], [], "The list of tool calls that failed from the last AI message."]
    
    # For structured output
    final_response: StructuredResponse
    
    
# --- PLAN & EXECUTE
# Inherit 'messages' key from MessagesState
class PlanExecuteMessagesState(MessagesState):
    """
    Messages extended for plan-execute agent.

    Tracks `plan`, accumulates `past_steps`, and holds the final (unstructured) `response`.
    Also includes `final_response` for structured output emission.
    """
    # For planning steps:
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add] # list of (Step, Result)
    response: str
    
    # For structured output
    final_response: StructuredResponse

# ---------------------------------------- PLAN-EXECUTE OUTPUTS ----------------------------------------    
# PLAN step: structured output
class Plan(TypedDict):
    """Plan to follow in future."""
    steps: Annotated[List[str], ..., "List of steps to follow, should be in sorted order. Each step should also contain a rationale."]

# REPLAN step: return_to_user and update_plan tools
# Tools are "dummy" placeholder tools -- it is the parameters that matter.
def return_to_user(response: str) -> None:
    """
    Returns the final response to the user. Use this tool if the plan is complete, no additional steps need to be taken, and a final answer can be returned to the user.
            
    Args:
        response (str): final response to return to the user.
    Returns:
        None
    """
    return

def update_plan(steps: list[str]) -> None:
    """
    Updates the plan with the provided steps. Use this tool if additional steps still need to be taken to complete the original task.  
            
    Args:
        steps (list[str]): list of remaining steps to do, e.g., ["next step 1", "next step 2", ...]. Each step should also contain a rationale.
    Returns:
        None
    """
    return


# ---------------------------------------- AGENT GRAPHS ----------------------------------------

class OllamaRetryHandler:
    """
    Retry helper for transient Ollama streaming errors.
    """
    def __init__(self, retries: int = 3, delay: float = 1.0):
        """
        Initialize the retry handler.

        Args:
            retries (int): Max retry attempts.
            delay (float): Delay between retries in seconds.
        """
        self.retries = retries
        self.delay = delay
    
    def is_retryable(self, e: Exception) -> bool:
        """
        Determine if an exception is retryable.

        Args:
            e (Exception): Exception thrown by Ollama.

        Returns:
            bool: True if the error message indicates a transient stream issue.
        """
        return isinstance(e, ValueError) and "No data received from Ollama stream" in str(e)
    
    def invoke_with_retry(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Invoke a function with retry on retryable errors.

        Args:
            fn (Callable[..., Any]): Function to invoke.
            *args: Positional args forwarded to `fn`.
            **kwargs: Keyword args forwarded to `fn`.

        Returns:
            Any: Result of the function call if successful.

        Raises:
            Exception: The last exception when retries are exhausted or non-retryable.
        """
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if self.is_retryable(e):
                    logging.warning(f"[Retry {attempt}] Retrying due to error: {e}")
                    time.sleep(self.delay)
                else:
                    raise
        logging.error("Max retries exceeded for retryable error.")
        raise
        
# Normalize tool call args across models (flat args vs. nested args["parameters"]).
# If a nested parameters chain exists, drill down to the deepest dict; otherwise return args as-is.
def extract_tool_args(tool_call):
    """
    Normalize tool call arguments.

    - If a nested `args["parameters"]` chain exists (each a dict), drills down
      and returns the deepest one.
    - Otherwise returns `args` as-is.

    Args:
        tool_call: Tool call object with `args` possibly containing `parameters` nesting.

    Returns:
        dict: Normalized arguments for tool invocation.
    """
    try:
        args = tool_call.get("args") or {}
    except Exception:
        return {}
    if not isinstance(args, dict):
        return {}

    curr = args
    while isinstance(curr.get("parameters"), dict):
        curr = curr["parameters"]
    return curr

def extract_tool_reasoning(tool_call):
    """
    Extract tool call reasoning from flat or nested parameters.

    Prefers `args["reasoning"]` when present; otherwise drills through nested
    `parameters` dicts to find a `reasoning` key.

    Args:
        tool_call: Tool call object with `args` possibly containing `parameters` nesting.

    Returns:
        str | None: Reasoning string if found, otherwise None.
    """
    try:
        args = tool_call.get("args") or {}
    except Exception:
        return None
    if isinstance(args, dict):
        # Prefer flat reasoning
        if args.get("reasoning"):
            return args.get("reasoning")
        # Drill through nested parameters to find reasoning
        curr = args
        while isinstance(curr, dict):
            if curr.get("reasoning"):
                return curr.get("reasoning")
            next_curr = curr.get("parameters")
            if not isinstance(next_curr, dict):
                break
            curr = next_curr
    return None

def initialize_simple_graph(model) -> CompiledStateGraph:
    """
    Initialize and compile a simple agent StateGraph with a single `agent` node.

    Args:
        model (BaseChatModel): Chat model to invoke for messages.

    Returns:
        CompiledStateGraph: Executable graph with a single agent.
    """
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    simple_graph = workflow.compile()
    return simple_graph

def initialize_simple_graph_structured_output(model) -> CompiledStateGraph:
    """
    Initialize and compile a simple agent graph with structured output.

    Has a single `agent` node producing a `final_response` via structured output.
    Modified from LangGraph documentation (Option 2): https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph_1

    Args:
        model (BaseChatModel): Model configured for structured output.

    Returns:
        CompiledStateGraph: Executable graph emitting `final_response`.
    """
    def call_model(state: CustomMessagesState):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [AIMessage(str(response))], "final_response": response}

    workflow = StateGraph(CustomMessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    simple_graph = workflow.compile()
    return simple_graph

def initialize_simple_graph_structured_output_two_step(model, model_with_structured_output, prompt_templates) -> CompiledStateGraph:
    """
    Initialize a two-step simple graph for structured output.

    First `agent` node produces unstructured output, then `respond` node
    composes a structured `final_response` using prompt templates.
    Modified from LangGraph documentation (Option 2): https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph_1

    Args:
        model (BaseChatModel): Base chat model for initial generation.
        model_with_structured_output (BaseChatModel): Model configured for structured output.
        prompt_templates (dict): Templates for system/human messages in `structured_output`.

    Returns:
        CompiledStateGraph: Executable graph emitting `final_response`.
    """
    retry_handler = OllamaRetryHandler()
    def call_model(state: CustomMessagesState):
        messages = state["messages"]
        response = retry_handler.invoke_with_retry(model.invoke, messages)
        return {"messages": [response]}
    
    def respond(state: CustomMessagesState):
        # Invoke the structured output LLM with relevant context + instructions
        messages = state["messages"]
        chat_history = ""
        for message in messages:
            chat_history += message.pretty_repr()
        
        response = retry_handler.invoke_with_retry(
            model_with_structured_output.invoke,
            [
                SystemMessage(content=prompt_templates["structured_output"]["system"]),
                HumanMessage(content=prompt_templates["structured_output"]["human"].format(messages=chat_history))
            ]
        )
        # Return the final answer
        # NOTE: this does NOT add the final response to the list of messages
        return {"final_response": response}
    
    workflow = StateGraph(CustomMessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("respond", respond)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "respond")
    workflow.add_edge("respond", END)

    checkpointer = MemorySaver()
    simple_graph = workflow.compile(checkpointer=checkpointer)
    return simple_graph

def initialize_react_graph(model_with_tools, tools) -> CompiledStateGraph:
    """
    Initialize and compile the ReAct agent graph.

    Modified from LangGraph documentation: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#react-agent

    Args:
        model_with_tools (BaseChatModel): Chat model bound to tools.
        tools (List[StructuredTool] | list): Tools available to the agent.

    Returns:
        CompiledStateGraph: Executable ReAct graph.
    """
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return END

    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    workflow = StateGraph(MessagesState)
    
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", END: END})
    # Any time a tool is called, we return to the agent to decide the next step
    workflow.add_edge("tools", "agent")

    react_graph = workflow.compile()
    return react_graph

def initialize_react_graph_with_tool_retry(model_with_tools: BaseChatModel, tools) -> CompiledStateGraph:
    """
    Initialize and compile ReAct graph with tool retry capability.

    Modified from LangGraph documentation: https://langchain-ai.github.io/langgraph/how-tos/tool-calling-errors/#custom-strategies

    Args:
        model_with_tools (BaseChatModel): Chat model bound to tools.
        tools (List[StructuredTool] | list): Tools available to the agent.

    Returns:
        CompiledStateGraph: Executable ReAct graph supporting retries.
    """
    retry_handler = OllamaRetryHandler()
    max_retries = 3
    
    def should_continue(state: CustomMessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        content_str = get_message_content_as_string(last_message.content)
        if ("final answer" not in content_str.lower()):
            return "agent"
        return END

    def call_model(state: CustomMessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.type == 'ai':
            added_instructions = [
                HumanMessage("Continue with your line of inquiry: you either haven't reached the 'Final Answer', or your previous tool calls could not be parsed. If the latter, try again with the same reasoning.")
            ]
        else:
            added_instructions = []
        response = retry_handler.invoke_with_retry(model_with_tools.invoke, messages + added_instructions)
        return {"messages": [response]}

    def call_tool(state: CustomMessagesState):
        tools_by_name = {tool.name: tool for tool in tools}
        messages = state["messages"]
        last_message = messages[-1]
        output_messages: list[ToolMessage]= []
        failed_tool_calls: list[ToolCall] = []

        for tool_call in last_message.tool_calls:
            try:
                tool_result = tools_by_name[tool_call["name"]].invoke(extract_tool_args(tool_call))
                output_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                # Capture the error inside ToolMessage for retry handling
                output_messages.append(
                    ToolMessage(
                        content = f"Tool call failed. Error: {str(e).replace(annoying_pydantic, '')}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs={"error": str(e)},
                    )
                )
                failed_tool_calls.append(tool_call)
        return {"messages": output_messages, "failed_tool_calls": failed_tool_calls}

    def should_retry(state: CustomMessagesState) -> Literal["agent", "retry_agent"]:
        failed_tool_calls: list[ToolCall] = state.get("failed_tool_calls", None)
    
        # Only retry if tool calls fom the last ai message failed
        if failed_tool_calls and len(failed_tool_calls) > 0:
            return "retry_agent"
        return "agent"

    def retry_tool_call(state: CustomMessagesState, retries_left = max_retries):
        messages = state["messages"]
        failed_tool_calls: list[ToolCall] = state["failed_tool_calls"]
        tools_by_name = [tool.name for tool in tools]

        # Construct retry message
        retry_prompt = f"The following tool call(s) failed to execute due to an incorrect tool name or arguments:\n"
        for tool_call in failed_tool_calls:
            reasoning = extract_tool_reasoning(tool_call) or '(reasoning not provided - you must ALWAYS provide your reasoning)'
            retry_prompt += f"Name: {tool_call['name']}, reasoning: {reasoning}\n"
        retry_prompt += "Try these tool call(s) using the same reasoning, but with the correct tool name or arguments.\n"
        retry_prompt += f"Recall, the tools you have access to are: {', '.join(tools_by_name)}. You must also always provide the rationale for calling each tool through the 'reasoning' argument."

        # Invoke the model with retry message
        retry_response: AIMessage = retry_handler.invoke_with_retry(model_with_tools.invoke, messages + [HumanMessage(retry_prompt)])
        
        if not (retry_response.tool_calls or retry_response.content) and (retries_left < max_retries):
            return retry_tool_call(state, retries_left - 1)
        
        # Add retry_response to list of messages, and reset failed_tool_calls to an empty list
        return {"messages": messages + [retry_response], "failed_tool_calls": []}

    workflow = StateGraph(CustomMessagesState)

    # Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)
    workflow.add_node("retry_agent", retry_tool_call)

    # Standard agent -> tool -> agent flow
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_conditional_edges("tools", should_retry)

    # Retry flow
    workflow.add_conditional_edges("retry_agent", should_continue)

    react_graph = workflow.compile()
    return react_graph

def initialize_react_graph_with_structured_output(model_with_tools, tools, model_with_structured_output, prompt_templates) -> CompiledStateGraph:
    """
    Initialize and compile ReAct graph with structured output.

    Modified from LangGraph documentation (Option 2): https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph_1

    Args:
        model_with_tools (BaseChatModel): Model bound to tools for ReAct flow.
        tools (List[StructuredTool] | list): Tools available to the agent.
        model_with_structured_output (BaseChatModel): Model emitting structured output.
        prompt_templates (dict): Templates for `structured_output` messages.

    Returns:
        CompiledStateGraph: Executable graph emitting `final_response`.
    """
    
    def call_model(state: CustomMessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: CustomMessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"
        else:
            # If no more tool calls, prepare structured final response
            return "respond"
    
    def respond(state: CustomMessagesState):
        # Invoke the structured output LLM with relevant context + instructions
        messages = state["messages"]
        chat_history = ""
        for message in messages:
            chat_history += message.pretty_repr()
            
        response = model_with_structured_output.invoke([
            SystemMessage(content=prompt_templates["structured_output"]["system"]),
            HumanMessage(content=prompt_templates["structured_output"]["human"].format(messages=chat_history))
        ])
        # Return the final answer
        # NOTE: this does NOT add the final response to the list of messages
        return {"final_response": response}
    
    workflow = StateGraph(CustomMessagesState)
    
    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    workflow.add_node("respond", respond)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "respond": "respond"})
    # Any time a tool is called, we return to the agent to decide the next step
    workflow.add_edge("tools", "agent")
    workflow.add_edge("respond", END)

    react_graph = workflow.compile()
    return react_graph

def initialize_react_graph_with_tool_retry_and_structured_output(model_with_tools: BaseChatModel, tools, model_with_structured_output: BaseChatModel, prompt_templates) -> CompiledStateGraph:
    """
    Initialize and compile ReAct graph with tool retry and structured output.

    Modified from LangGraph documentation: https://langchain-ai.github.io/langgraph/how-tos/tool-calling-errors/#custom-strategies

    Args:
        model_with_tools (BaseChatModel): Model bound to tools for ReAct flow.
        tools (List[StructuredTool] | list): Tools available to the agent.
        model_with_structured_output (BaseChatModel): Model emitting structured output.
        prompt_templates (dict): Templates for `structured_output` messages.

    Returns:
        CompiledStateGraph: Executable graph emitting `final_response` with retry support.
    """
    retry_handler = OllamaRetryHandler()
    max_retries = 3
    
    def should_continue(state: CustomMessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        content_str = get_message_content_as_string(last_message.content)
        if ("final answer" not in content_str.lower()):
            return "agent"
        return "respond"

    def call_model(state: CustomMessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.type == 'ai':
            added_instructions = [
                HumanMessage("Continue with your line of inquiry: you either haven't reached the 'Final Answer', or your previous tool calls could not be parsed. If the latter, try again with the same reasoning.")
            ]
        else:
            added_instructions = []
        response = retry_handler.invoke_with_retry(model_with_tools.invoke, messages + added_instructions)
        return {"messages": [response]}

    def call_tool(state: CustomMessagesState):
        tools_by_name = {tool.name: tool for tool in tools}
        messages = state["messages"]
        last_message = messages[-1]
        output_messages: list[ToolMessage]= []
        failed_tool_calls: list[ToolCall] = []

        for tool_call in last_message.tool_calls:
            try:
                tool_result = tools_by_name[tool_call["name"]].invoke(extract_tool_args(tool_call))
                output_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                # Capture the error inside ToolMessage for retry handling
                output_messages.append(
                    ToolMessage(
                        content = f"Tool call failed. Error: {str(e).replace(annoying_pydantic, '')}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs={"error": str(e)},
                    )
                )
                failed_tool_calls.append(tool_call)
        return {"messages": output_messages, "failed_tool_calls": failed_tool_calls}

    def should_retry(state: CustomMessagesState) -> Literal["agent", "retry_agent"]:
        failed_tool_calls: list[ToolCall] = state.get("failed_tool_calls", None)
    
        # Only retry if tool calls from the last ai message failed
        if failed_tool_calls and len(failed_tool_calls) > 0:
            return "retry_agent"
        return "agent"

    def retry_tool_call(state: CustomMessagesState, retries_left = max_retries):
        messages = state["messages"]
        failed_tool_calls: list[ToolCall] = state["failed_tool_calls"]
        tools_by_name = [tool.name for tool in tools]

        # Construct retry message
        retry_prompt = f"The following tool call(s) failed to execute due to an incorrect tool name or arguments:\n"
        for tool_call in failed_tool_calls:
            reasoning = extract_tool_reasoning(tool_call) or '(reasoning not provided - you must ALWAYS provide your reasoning)'
            retry_prompt += f"Name: {tool_call['name']}, reasoning: {reasoning}\n"
        retry_prompt += "Try these tool call(s) using the same reasoning, but with the correct tool name or arguments.\n"
        retry_prompt += f"Recall, the tools you have access to are: {', '.join(tools_by_name)}. You must also always provide the rationale for calling each tool through the 'reasoning' argument."

        # Invoke the model with retry message
        retry_response: AIMessage = retry_handler.invoke_with_retry(model_with_tools.invoke, messages + [HumanMessage(retry_prompt)])
        
        if not (retry_response.tool_calls or retry_response.content) and (retries_left < max_retries):
            return retry_tool_call(state, retries_left - 1)
        
        # Add retry_response to list of messages, and reset failed_tool_calls to an empty list
        return {"messages": messages + [retry_response], "failed_tool_calls": []}
    
    def respond(state: CustomMessagesState):
        # Invoke the structured output LLM with relevant context + instructions
        messages = state["messages"]
        chat_history = ""
        for message in messages:
            chat_history += message.pretty_repr()
            
        response = retry_handler.invoke_with_retry(
            model_with_structured_output.invoke,
            [
                SystemMessage(content=prompt_templates["structured_output"]["system"]),
                HumanMessage(content=prompt_templates["structured_output"]["human"].format(messages=chat_history))
            ]
        )
        # Return the final answer
        # NOTE: this does NOT add the final response to the list of messages
        return {"final_response": response}

    workflow = StateGraph(CustomMessagesState)

    # Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)
    workflow.add_node("retry_agent", retry_tool_call)
    workflow.add_node("respond", respond)

    # Standard agent -> tool -> agent flow
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_conditional_edges("tools", should_retry)

    # Retry flow
    workflow.add_conditional_edges("retry_agent", should_continue)
    
    # Add respond edge
    workflow.add_edge("respond", END)

    checkpointer = MemorySaver()
    react_graph = workflow.compile(checkpointer=checkpointer)
    return react_graph


def initialize_plan_execute_react_shared_state(model: BaseChatModel, model_with_tools: BaseChatModel, tools, model_with_structured_output: BaseChatModel, prompt_templates) -> CompiledStateGraph:
    """
    Initialize plan-execute ReAct graph with shared state and structured output.

    Composes a planner (`Plan` structured output), an executor agent with tool
    retry, a replanner using tools (`return_to_user`, `update_plan`), and a
    `respond` step producing structured output.

    Args:
        model (BaseChatModel): Base chat model used for planner/replanner.
        model_with_tools (BaseChatModel): ReAct executor model bound to tools.
        tools (List[StructuredTool] | list): Tools available to the executor.
        model_with_structured_output (BaseChatModel): Model emitting structured output responses.
        prompt_templates (dict): Planner, replanner, execute, and structured output templates.

    Returns:
        CompiledStateGraph: Executable plan-execute graph with shared state.
    """
    executor_agent = initialize_react_graph_with_tool_retry(model_with_tools, tools)
    retry_handler = OllamaRetryHandler()
    
    # Prompt templates for planner and replanner
    planner_prompt_template = prompt_templates["plan_execute"]["planner"]
    replanner_prompt_template = prompt_templates["plan_execute"]["replanner"]
    
    ## Setup planner and replanner models
    planner = model.with_structured_output(Plan)
    replanner_tools = [
        StructuredTool.from_function(return_to_user, return_direct=True),
        StructuredTool.from_function(update_plan, return_direct=True)
    ] 
    replanner = model.bind_tools(replanner_tools) # model.with_structured_output(Replan)
    
    max_retries = 5
    def plan_step(state: PlanExecuteMessagesState, retries_left = max_retries):
        """Initial planning step - generates a plan based on user input."""
        messages = state['messages']
        planner_instruction = HumanMessage(content=planner_prompt_template)
        
        plan_response: Plan = retry_handler.invoke_with_retry(planner.invoke, messages + [planner_instruction])
        if not plan_response:
            if retries_left > 0:
                return plan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Planner output is none at {max_retries} tries.")
        if not plan_response.get("steps", False):
            if retries_left > 0:
                return plan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Planner output does not contain steps at {max_retries} tries.")
        if len(plan_response['steps']) == 0:
            if retries_left > 0:
                return plan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Planner output contains empty steps list at {max_retries} tries.")            
            
        # Optionally add to messages: "messages": [planner_instruction, AIMessage(str(plan_response))]}
        return {"plan": plan_response["steps"]}

    def execute_step(state: PlanExecuteMessagesState):
        """Executes the next step in the plan using the agent"""
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0] # Take the top step

        human_message_template = prompt_templates["plan_execute"]["execute"]["human"]
        
        messages = state["messages"]
        execute_next_step_message = HumanMessage(content=human_message_template.format(plan=plan_str, task=task))
        response = executor_agent.invoke({"messages": messages + [execute_next_step_message]})
        
        # This will include the execute_next_step_message as the first message
        new_messages = response["messages"][len(messages):]
        
        response_last_message: BaseMessage = response['messages'][-1]
        if response_last_message and response_last_message.type == "ai":
            task_result = get_message_content_as_string(response_last_message.content)
        else:
            task_result = ""
        return {"past_steps": [(task, task_result)], "messages": new_messages}
    
    def replan_step(state: PlanExecuteMessagesState, retries_left = max_retries):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        messages = state["messages"]
        past_steps = [f"- {instruction}. {retrieve_non_think_content(result)}" for instruction, result in state["past_steps"]]
        past_steps_str = "\n".join(past_steps) if len(past_steps) > 0 else "None"
        
        replan_instruction = HumanMessage(content=replanner_prompt_template.format(
            plan=plan_str, past_steps=past_steps_str))
        
        replan_response: AIMessage = retry_handler.invoke_with_retry(replanner.invoke, messages + [replan_instruction])
        if not replan_response or (len(replan_response.tool_calls) == 0):
            if retries_left > 0:
                return replan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Replanner output is none at {max_retries} tries.")
        if len(replan_response.tool_calls) > 1:
            if retries_left > 0:
                return replan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Replanner output called multiple tool calls at {max_retries} tries.")
        
        tool_call: ToolCall = replan_response.tool_calls[0]
        if tool_call['name'] == update_plan.__name__:
            try:
                args = extract_tool_args(tool_call)
                return {"plan": args['steps']}
            except Exception:
                if retries_left > 0:
                    return replan_step(state, retries_left - 1)
                else:
                    raise ValueError(f"Replanner output did not have 'steps' in update_plan arguments at {max_retries} tries.")
        elif tool_call['name'] == return_to_user.__name__:
            try:
                args = extract_tool_args(tool_call)
                return {"response": args['response']}
            except Exception:
                if retries_left > 0:
                    return replan_step(state, retries_left - 1)
                else:
                    raise ValueError(f"Replanner output did not have 'response' in return_to_user arguments at {max_retries} tries.")
        else:
            if retries_left > 0:
                return replan_step(state, retries_left - 1)
            else:
                raise ValueError(f"Invalid replanner output: {tool_call}") 
    
    def should_end(state: PlanExecuteMessagesState):
        """Determines if execution should stop"""
        if "response" in state and state["response"]:
            return "respond"
        else:
            return "agent"
        
    def respond(state: PlanExecuteMessagesState):
        # Invoke the structured output LLM with relevant context + instructions
        messages: List[BaseMessage] = state["messages"]
        
        # The whole chat history is likely too long. Only include most relevant messages. 
        chat_history = ""
        for message in messages:
            if (message.type == "system") or (message.type == "human") or (message.type == "ai" and message.content):
                chat_history += message.pretty_repr()
            
        response = retry_handler.invoke_with_retry(
            model_with_structured_output.invoke,
            [
                SystemMessage(content=prompt_templates["structured_output"]["system"]),
                HumanMessage(content=prompt_templates["structured_output"]["human"].format(messages=chat_history))
            ]
        )
        # Return the final answer
        # NOTE: this does NOT add the final response to the list of messages
        return {"final_response": response}
    
    workflow = StateGraph(PlanExecuteMessagesState)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("respond", respond)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_end, ["agent", "respond"])
    workflow.add_edge("respond", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph

def draw_langgraph_graph(graph: CompiledStateGraph):
    """
    Display a compiled LangGraph as a Mermaid PNG in compatible environments.

    Args:
        graph (CompiledStateGraph): Graph to visualize.

    Returns:
        None
    """
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    