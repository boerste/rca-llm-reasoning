import os
import ast
import pandas as pd
from pandas.api.types import is_scalar
from typing import List, Union

def is_none(x):
    """
    Return True for None, empty iterables, or iterables/series where all elements are NA.

    Args:
        x: The input value to check.

    Returns:
        bool: True if the input is effectively None/empty, False otherwise.
    """
    if x is None:
        return True
    # scalar (including strings) -> use pandas isna
    if is_scalar(x):
        return pd.isna(x)
    # non-scalar: empty container -> treat as None
    if hasattr(x, "__len__"):
        try:
            if len(x) == 0:
                return True
        except Exception:
            pass
    # check if all elements are NA (works for lists, tuples, Series, numpy arrays)
    try:
        return all(pd.isna(v) for v in x)
    except Exception:
        return False
    
def safe_eval(x):
    """
    Safely evaluate a string representation to a Python object.

    For strings that represent sequences (lists/tuples) or single values, this function
    will return a list of strings. Returns None for values considered
    "none" or the empty string. Tries ast.literal_eval first, then pandas.eval, then
    falls back to returning the original value wrapped as a single-item list of str.

    Args:
        x: The input value to evaluate (ideally a string).

    Returns:
        list[str] | None: A list of strings representing the evaluated content, or None.
    """
    if is_none(x) or x == "":
        return None
    # If already a list/tuple -> coerce items to str and return
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [str(i) for i in v]
            return [str(v)]
        except Exception:
            try:
                v = pd.eval(x)
                if isinstance(v, (list, tuple)):
                    return [str(i) for i in v]
                return [str(v)]
            except Exception:
                # fallback: return the raw string as single-item list
                return [x]
    # Non-string, non-list input: return as-is
    return x

def find_key_recursively(data: Union[dict, list], target_key: str) -> List[dict]:
    """
    Recursively search for all occurrences of a specific key in a nested dictionary or list.
    
    Args:
        data: The input data (dict or list).
        target_key: The key to search for.

    Returns:
        A list of values associated with the target key.
    """
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key and isinstance(value, list):
                results.extend(value)
            else:
                results.extend(find_key_recursively(value, target_key))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_key_recursively(item, target_key))
    return results

def get_kg_id_from_file_name(kg_file: str) -> str:
    """
    Extract the knowledge graph ID from a given filename.
    
    Args:
        kg_file (str): The filename or path of the knowledge graph JSON file.
        
    Returns:
        str: The ID extracted from the filename (removing the extension).
    """
    kg_basename = os.path.basename(kg_file)
    return kg_basename.split('.json')[0]
    
def retrieve_non_think_content(response: str) -> str:
    """
    Extracts and returns the part of the response that comes after the last </think> tag.
    If no <think> tags are found, returns the original response.

    Args:
        response (str): The raw response string that may contain <think> tags.

    Returns:
        str: The content after reasoning tokens, or the original response.
    """
    # Check to see if response contains think-tags: <think> and </think>
    if ("<think>" in response) and ("</think>" in response):
        try:
            # Return everyting after the last </think> tag
            result = response.split("</think>", 1)[1]
            return result.lstrip()
        except Exception:
            # Failed on string split, return response as-is
            return response
    return response

def get_json_schema_from_typeddict(typeddict_class) -> dict:
    """Converts a TypedDict class to a JSON schema with additionalProperties: false.

    This is needed for OpenRouter/Anthropic which requires strict JSON schemas.

    Args:
        typeddict_class: A TypedDict class (e.g., RootCausesResponse)

    Returns:
        A JSON schema dictionary compatible with Anthropic's structured output API
    """
    from typing import get_type_hints, get_origin, get_args
    import typing

    def python_type_to_json_schema(python_type, annotations=None) -> dict:
        """
        Convert a Python type to JSON schema.
        
        Args:
            python_type: The Python type to convert.
            annotations: Optional annotations (unused in current implementation but kept for signature).
        
        Returns:
            dict: The corresponding JSON schema dictionary.
        """
        origin = get_origin(python_type)

        # Handle Annotated types (extract the base type)
        if origin is typing.Annotated:
            args = get_args(python_type)
            base_type = args[0]
            # Get description from annotations if available
            description = None
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break
            schema = python_type_to_json_schema(base_type)
            if description:
                schema["description"] = description
            return schema

        # Handle List types
        if origin is list:
            args = get_args(python_type)
            item_type = args[0] if args else typing.Any
            return {
                "type": "array",
                "items": python_type_to_json_schema(item_type)
            }

        # Handle basic types
        if python_type is str:
            return {"type": "string"}
        if python_type is int:
            return {"type": "integer"}
        if python_type is float:
            return {"type": "number"}
        if python_type is bool:
            return {"type": "boolean"}
        if python_type is type(None):
            return {"type": "null"}

        # Handle TypedDict (nested)
        if hasattr(python_type, "__annotations__"):
            return typeddict_to_schema(python_type)

        # Default fallback
        return {"type": "string"}

    def typeddict_to_schema(td_class, is_root: bool = False) -> dict:
        """
        Convert a TypedDict to JSON schema.
        
        Args:
            td_class: The TypedDict class to convert.
            is_root (bool): Whether this is the root object schema.

        Returns:
            dict: The JSON schema for the TypedDict.
        """
        hints = get_type_hints(td_class, include_extras=True)

        properties = {}
        required = []

        for field_name, field_type in hints.items():
            properties[field_name] = python_type_to_json_schema(field_type)
            required.append(field_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

        # Add title (required by LangChain for root schema)
        if is_root:
            schema["title"] = td_class.__name__

        # Add description from docstring if available
        if td_class.__doc__:
            schema["description"] = td_class.__doc__

        return schema

    return typeddict_to_schema(typeddict_class, is_root=True)
