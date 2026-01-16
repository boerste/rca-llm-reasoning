from typing_extensions import Annotated, TypedDict, List

# TypedDict (see https://python.langchain.com/docs/how_to/structured_output/#typeddict-or-json-schema)

class StructuredResponse(TypedDict):
    pass
    
class RootCause(TypedDict):
    """A root cause of the observed symptoms."""
    type: Annotated[str, ..., "The type of root cause fault."]
    description: Annotated[str, ..., "An explanation of the root cause fault."]
    location: Annotated[str, ..., "The exact node or edge at which the root cause fault occurs."]
    justification: Annotated[str, ..., "A step-by-step reasoning, grounded in the system knowledge graph, explaining how the symptoms could occur due to the root cause."]
    propagation_path: Annotated[str, ..., "The specific propagation path in the knowledge graph that would make the root cause possible, formatted as node1 --(edge_label1)--> node2 --(edge_label2)--> node3."]

class RootCausesResponse(StructuredResponse):
    """The possible root causes for the given symptoms."""
    root_causes: Annotated[List[RootCause], [], "The list of root causes."]