import os
import json
import ast
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.load import dumps, loads
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Union, Any, Iterator, Optional
from pprint import pprint

from utils.common import safe_eval, get_json_schema_from_typeddict

def add_additional_properties_false(schema: dict) -> dict:
    """Recursively adds 'additionalProperties': false to all object types in a JSON schema.

    This is required for Anthropic's structured output API which requires strict schemas.
    OpenRouter passes through to Anthropic, so it has the same requirement.

    Args:
        schema: A JSON schema dictionary

    Returns:
        The modified schema with additionalProperties: false on all objects
    """
    if not isinstance(schema, dict):
        return schema

    # Create a copy to avoid mutating the original
    schema = schema.copy()

    # If this is an object type, add additionalProperties: false
    if schema.get("type") == "object":
        schema["additionalProperties"] = False

        # Process nested properties
        if "properties" in schema:
            schema["properties"] = {
                k: add_additional_properties_false(v)
                for k, v in schema["properties"].items()
            }

    # Handle array items
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = add_additional_properties_false(schema["items"])

    # Handle anyOf, allOf, oneOf
    for key in ("anyOf", "allOf", "oneOf"):
        if key in schema:
            schema[key] = [add_additional_properties_false(item) for item in schema[key]]

    # Handle definitions/$defs
    for key in ("definitions", "$defs"):
        if key in schema:
            schema[key] = {
                k: add_additional_properties_false(v)
                for k, v in schema[key].items()
            }

    return schema

@dataclass
class ModelResult:
    """
    Represents the model result of a single fault scenario execution.

    Attributes:
        kg_id (str): ID of the knowledge graph used.
        fault_id (str): ID of the fault scenario being analyzed.
        messages (List[BaseMessage]): The message history leading to the result.
        final_response (dict | None): The structured final response from the model.
        ttr (float): Time to result (execution duration).
        error (list): List of any errors encountered during execution.
        past_steps (list | None): Sequence of past steps (used for plan-execute agents).
    """
    kg_id: str
    fault_id: str
    messages: List[BaseMessage]
    final_response: dict | None
    ttr: float
    error: list
    past_steps: list | None = None # for plan-execute only

class ExecutedToolCall:
    """Represents an executed tool call in a message chain."""
    def __init__(self, name: str, args: dict[str, Any], id: Optional[str] = None,
                 status: str = "not_run", content: Optional[str] = None, execution_order: int = -1):
        self.name = name
        self.args = args
        self.id = id
        self.status = status
        self.content = content
        self.execution_order = execution_order

    def __repr__(self):
        # return (f"ExecutedToolCall(name={self.name}, args={self.args}, id={self.id}, "
        #         f"status={self.status}, content={self.content}, execution_order={self.execution_order})")
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_string(cls, json_str: str):
        return cls(**json.loads(json_str))

def load_model_ollama(model_name = 'llama3.1:8b', max_new_tokens = 2048, num_ctx = 14848, reasoning=None) -> ChatOllama:
    """
    Loads model as a Langchain model from Ollama.
    Note: this requires the model to be pulled from Ollama first. 
    - Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    - Pull model: ollama pull [model_name, e.g., llama3.2:3b]

    Args:
        model_name: Ollama model name

    Returns:
        Langchain chat model from Ollama 
    """    
    # Set max_new_tokens to num_ctx if -2
    if max_new_tokens == -2:
        max_new_tokens = num_ctx
        
    print(f"Loading {model_name} model using Langchain ChatOllama with max_new_tokens={max_new_tokens}, num_ctx={num_ctx}...")
    
    if any(s in model_name.lower() for s in ("qwen3", "deepseek-r1")) and reasoning:
        is_reasoning = True
    else:
        is_reasoning= None
        
    model = ChatOllama(
        model=model_name,
        num_predict=max_new_tokens,
        repeat_last_n=-1,
        num_ctx=num_ctx,
        disable_streaming='tool_calling',
        cache=False,
        reasoning=is_reasoning,
        )

    return model

def load_model_openrouter(model_name: str, max_new_tokens: int = 2048, num_ctx: int = None,
                          reasoning_effort: str = None, reasoning_max_tokens: int = None) -> ChatOpenAI:
    """Loads model via OpenRouter API (supports Claude, Kimi, MiniMax, and many others).
    Note: this requires the OPENROUTER_API_KEY environment variable to be set.

    Args:
        model_name: OpenRouter model ID. Examples:
            - anthropic/claude-opus-4-5-20250514
            - anthropic/claude-sonnet-4-5-20241022
            - moonshotai/kimi-k2
            - minimax/minimax-m2
        max_new_tokens: Maximum tokens for response generation
        num_ctx: Unused, kept for interface compatibility with load_model_ollama
        reasoning_effort: Reasoning effort level for hybrid models like Claude Opus 4.5.
            Options: "low", "medium", "high", or None (model decides, usually medium)
        reasoning_max_tokens: Max tokens for reasoning/thinking. Use this to control
            costs on expensive models. If set, overrides the effort level.

    Returns:
        Langchain chat model via OpenRouter
    """
    # Handle -2 convention (use large default)
    if max_new_tokens == -2:
        max_new_tokens = 8192  # Reasonable default for extended responses

    # Build extra_body for reasoning parameters
    extra_body = {}
    if reasoning_max_tokens is not None:
        extra_body["reasoning"] = {"max_tokens": reasoning_max_tokens}
        print(f"Loading {model_name} model using OpenRouter with max_tokens={max_new_tokens}, reasoning_max_tokens={reasoning_max_tokens}...")
    elif reasoning_effort is not None:
        extra_body["reasoning"] = {"effort": reasoning_effort}
        print(f"Loading {model_name} model using OpenRouter with max_tokens={max_new_tokens}, reasoning_effort={reasoning_effort}...")
    else:
        print(f"Loading {model_name} model using OpenRouter with max_tokens={max_new_tokens}...")

    model = ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        max_tokens=max_new_tokens,
        temperature=0,
        model_kwargs=extra_body if extra_body else {},
    )
    return model

def is_openrouter_model(model: BaseChatModel) -> bool:
    """
    Check if the model is an OpenRouter model by checking the base URL.

    Args:
        model (BaseChatModel): The chat model to check.

    Returns:
        bool: True if the model is configured with an OpenRouter compatible base URL.
    """
    if hasattr(model, 'openai_api_base'):
        return 'openrouter.ai' in str(model.openai_api_base or '')
    return False

def get_model_with_structured_output(model: BaseChatModel, structured_output_class):
    """
    Configure a model to output structured data according to a schema.
    
    Compatible with both standard models (via TypedDict) and OpenRouter/Anthropic models 
    (which require strict JSON schema validation).

    Args:
        model (BaseChatModel): The base chat model.
        structured_output_class: A TypedDict class defining the expected output structure.

    Returns:
        Runnable: A runnable model configured to return structured output.
    """
    if is_openrouter_model(model):
        # OpenRouter/Anthropic requires additionalProperties: false in JSON schemas
        json_schema = get_json_schema_from_typeddict(structured_output_class)
        return model.with_structured_output(schema=json_schema)
    else:
        # Ollama and other models work fine with TypedDict classes directly
        return model.with_structured_output(schema=structured_output_class)

def print_stream_chunk(s, printed_count = 0):
    """
    Prints a single graph execution step (chunk) from a message stream.
    Handles 'final_response' chunks by pretty-printing and 'messages' chunks by 
    printing new messages since the last count.

    Args:
        s (dict): A single stream chunk/step.
        printed_count (int): Number of messages printed so far in the stream. Defaults to 0.

    Returns:
        int | None: The new count of printed messages if 'messages' was in the chunk, else None.

    Raises:
        ValueError: If the chunk must contain 'messages' or 'final_response' but has neither.
    """
    if "final_response" in s:
        # case for react_graph structured outputs
        final_response = s["final_response"]
        pprint(final_response, width=200)
    elif "messages" in s:
        new_messages = s["messages"][printed_count:]
        for message in new_messages:
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
        return len(s["messages"])
    else:
        raise ValueError("Unsupported chunk structure: missing 'messages' or 'final_response'")

def print_stream(stream: Iterator[Union[dict[str, Any], Any]]):
    """
    Prints messages from a stream of graph execution steps.

    Args:
        stream (Iterator): An iterator yielding execution steps (chunks).
        
    Returns:
        Any: The last item processed from the stream.
    """
    printed_count = 0 # tracks how many messages have been printed so far
    for s in stream:
        val = print_stream_chunk(s, printed_count)
        if val is not None:
            printed_count = val      

def load_messages_from_string(messages_str: str) -> List[BaseMessage]:
    """
    Loads Langchain messages from a JSON string using Langchain's load mechanism.
    
    Includes error handling for "not_implemented" messages which can occur 
    in response metadata.

    Args:
        messages_str (str): String format of list of messages (created by dumps(messages)).

    Returns:
        List[BaseMessage]: A list of instantiated BaseMessage objects.
    """
    try:    
        messages_obj = json.loads(messages_str)
    except Exception:
        return messages_str
    
    # messages holds instantiated BaseMessages
    messages: List[BaseMessage] = []
    
    for i in range(len(messages_obj)):
        message_obj = messages_obj[i]
        try:
            # Try to deserialize the message using the langchain loads function
            message = loads(json.dumps(message_obj))
            messages.append(message)
        except Exception as e:
            # print(f"Error loading message {i}: {e}")
            
            # The error usually lies in the "kwargs">"response_metadata">"message" field
            # check if this problematic field exists
            if (
                "kwargs" in message_obj
                and "response_metadata" in message_obj["kwargs"]
                and "message" in message_obj["kwargs"]["response_metadata"]
            ):
                # Temporarily remove the problematic field
                problematic_field = message_obj["kwargs"]["response_metadata"]["message"]
                del message_obj["kwargs"]["response_metadata"]["message"]
                
                # Retry deserialization without the problematic field
                try:
                    message = loads(json.dumps(message_obj))
                    # Add problematic field back into deserialized message
                    message.response_metadata['message'] = problematic_field
                    messages.append(message)
                    
                except Exception as retry_e:
                    print(f"Still couldn't process message{i} after adjustment: {retry_e}")
                    
    return messages

def load_messages_from_string_openai(body_choices_str):
    """
    Load messages from OpenAI response body choices string.
    
    Parses the string representation of choices from an OpenAI API response
    into a list of message objects.

    Args:
        body_choices_str (str | list): String representation of the choices list or the list itself.

    Returns:
        list: A list of message dictionaries or objects containing the message content.
    """
    messages = []
    if isinstance(body_choices_str, str):
        body_choices = safe_eval(body_choices_str)
    else:
        body_choices = body_choices_str
        
    for message_str_list_str in body_choices:
        message_str_list = safe_eval(message_str_list_str)
        if isinstance(message_str_list, list):
            message_str = message_str_list[0]
        else:
            message_str = message_str_list
        if isinstance(message_str, str):
            message = ast.literal_eval(message_str)
        else:
            message = message_str
        messages.append(message['message'] if 'message' in message else message)
    return messages

def get_message_content_as_string(content) -> str:
    """
    Extract message content as a string.

    Handles both string content (Ollama) and list content (Anthropic/OpenAI).
    Anthropic models return content as a list of content blocks for multimodal support.

    Args:
        content (str | list): The content payload from a message, either a string or list of blocks.

    Returns:
        str: The extracted text content joined into a single string.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Join text from all content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    return str(content)