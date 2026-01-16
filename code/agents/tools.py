
import json
import networkx as nx
from langchain_core.tools import StructuredTool
from typing_extensions import List
from dataclasses import asdict
from utils.kg import GraphEntity, GraphRelationship

# ---------------------------------------- TOOL HELPERS ----------------------------------------
def convert_edges_to_list_string(graph: nx.MultiGraph | nx.DiGraph, edges, is_reverse: bool) -> str:
    """
    Format edges as a labeled list string.

    Args:
        graph (nx.MultiGraph | nx.DiGraph): Graph containing edge attributes.
        edges (Iterable[Tuple[str, str]]): Edge tuples `(u, v)` to format.
        is_reverse (bool): Whether to reverse each edge `(u, v)` to `(v, u)` before labeling.

    Returns:
        str: Labeled edges, one per line, e.g., `- A --(label)--> B`.
    """
    # Retrieve edge labels
    labeled_edges = []
    if is_reverse:
        edges = [(v, u) for u, v in edges]
    
    for u, v in edges:
        edge_data = graph.get_edge_data(u,v)
        
        # If there are multiple edges between nodes, retrieve both edge labels
        if graph.is_multigraph():
            labels = [f"{data.get('label', 'No label')}" for key, data in edge_data.items()]
            labeled_edges.append(f"{u} --({', '.join(labels)})--> {v}")
        else:
            label = edge_data.get('label', 'No label')
            labeled_edges.append(f"{u} --({label})--> {v}")
    edges_str = "\n".join([f"- {edge}" for edge in labeled_edges])
    return edges_str

def convert_edges_to_json_string(graph: nx.MultiGraph | nx.DiGraph, edges, is_reverse: bool) -> str:
    """
    Format edges as a JSON code block string.

    Produces a fenced JSON block containing `edges` objects with `source`,
    `target`, and `label` (list or string depending on graph type).

    Args:
        graph (nx.MultiGraph | nx.DiGraph): Graph containing edge attributes.
        edges (Iterable[Tuple[str, str]]): Edge tuples `(u, v)` to format.
        is_reverse (bool): Whether to reverse each edge `(u, v)` to `(v, u)` before labeling.

    Returns:
        str: JSON code block string with an `edges` array.
    """
    if is_reverse:
        edges = [(v, u) for u, v in edges]
    
    labeled_edges = []
    
    for u, v in edges:
        edge_data = graph.get_edge_data(u,v)
        
        # If there are multiple edges between nodes, retrieve both edge labels
        if graph.is_multigraph():
            labels = [f"{data.get('label', 'No label')}" for key, data in edge_data.items()]
        else:
            labels = edge_data.get('label', 'No label')
        
        labeled_edges.append({
            "source": u,
            "target": v,
            "label": labels
        })
    edges_str = f"```json\n{json.dumps({"edges": labeled_edges}, indent=None)}\n```"
    return edges_str

def convert_nodes_to_list_string(nodes, with_type: bool) -> str:
    """
    Format nodes as a list string.

    Args:
        nodes (Iterable): Nodes, optionally with `(node, data)` tuples when `with_type=True`.
        with_type (bool): Whether to include the node's `type` in the output.

    Returns:
        str: Labeled nodes, one per line, e.g., `- service` or `- service (Cache)`.
    """
    if with_type:
        nodes_info = []
        for node, data in nodes:
            nodes_info.append(f"{node} ({data['type']})")
        return "\n".join(f"- {n}" for n in nodes_info)
    else:
        return "\n".join(f"- {n}" for n in nodes)
    
def convert_nodes_to_json_string(nodes, with_type: bool) -> str:
    """
    Format nodes as a JSON code block string.

    Args:
        nodes (Iterable): Nodes, optionally with `(node, data)` tuples when `with_type=True`.
        with_type (bool): Whether to include the node's `type` in the output.

    Returns:
        str: JSON code block string with `nodes` array.
    """
    if with_type:
        nodes_info = [{"name": node, "type": data['type']} for node, data in nodes]
        return f"```json\n{json.dumps({"nodes": nodes_info}, indent=None)}\n```"
    else:
        return f"```json\n{json.dumps({"nodes": list(nodes)}, indent=None)}\n```"
        
def format_attributes(attrs):
    """
    Format an attribute dict into a human-readable multi-line string.

    Lists are expanded into indented bullets; scalars are shown as `key: value`.

    Args:
        attrs (dict): Attributes dictionary to format.

    Returns:
        str: Multi-line formatted attributes.
    """
    lines = []
    for k, v in attrs.items():
        if isinstance(v, list):
            lines.append(f"{k}:" if v else f"{k}: []")
            lines.extend(f"  - {item}" for item in v)
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
 
def get_edge_attributes_helper(source, target, graph) -> str:
    """
    Return all attributes of the edge(s) between `source` and `target`.

    Args:
        source (str): Source node id.
        target (str): Target node id.
        graph (nx.MultiDiGraph): Graph containing edge attributes.

    Returns:
        str: Formatted attributes per edge key, including labels and non-label fields.
    """
    edges = graph.get_edge_data(source, target)
    results = []
    for key, attr in edges.items():
        label = attr.get('label', "")
        filtered_attr = {k: v for k, v in attr.items() if k != 'label'}
        attr_str = format_attributes(filtered_attr) or "  - None (no alerts were detected)."
        results.append(f"{key + 1}. `{source} --({label})--> {target}`:\n{attr_str}")

    results_str = '\n'.join(results)
    return results_str

def get_all_edges(graph: nx.MultiDiGraph, strategy: str) -> str:
    """
    Return all edges in the graph as a string.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.
        strategy (str): Representation strategy: `list` or `json`.

    Returns:
        str: Edges formatted according to `strategy`.

    Raises:
        ValueError: If `strategy` is not `list` or `json`.
    """
    edges = graph.edges()
    if strategy == "list":
        return convert_edges_to_list_string(graph, edges, is_reverse=False)
    elif strategy == "json":
        return convert_edges_to_json_string(graph, edges, is_reverse=False)
    else:
        raise ValueError("Strategy must be one of 'list' or 'json'")

def get_all_nodes(graph: nx.MultiDiGraph, strategy: str) -> str:
    """
    Return all nodes in the graph as a string.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.
        strategy (str): Representation strategy: `list` or `json`.

    Returns:
        str: Nodes formatted according to `strategy`.

    Raises:
        ValueError: If `strategy` is not `list` or `json`.
    """
    nodes = graph.nodes(data=True)
    if strategy == "list":
        return convert_nodes_to_list_string(nodes, with_type=True)
    elif strategy == "json":
        return convert_nodes_to_json_string(nodes, with_type=True)
    else:
        raise ValueError("Strategy must be one of 'list' or 'json'")

def get_node_types_in_graph(graph) -> List[str]:
    """
    Retrieve all unique entity types present in the graph.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.

    Returns:
        List[str]: Unique `type` values across nodes.
    """
    node_types = set()
    for (node, nodedata) in graph.nodes.items():
        node_types.add(nodedata['type'])
    return node_types

def get_edge_types_in_graph(graph) -> List[str]:
    """
    Retrieve all unique relationship labels present in the graph.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.

    Returns:
        List[str]: Unique edge `label` values.
    """
    edge_types = set()
    for (u, v, edgedata) in graph.edges(data=True):
        edge_types.add(edgedata['label'])
    return edge_types
    
def get_schema_of_node_type(node_type: str, node_schemas) -> GraphEntity:
    """
    Retrieve the entity schema from the E-R specification given a node type.

    A node type can match the entity `name` or any of its `examples`.

    Args:
        node_type (str): Node type identifier (underscores allowed).
        node_schemas (List[GraphEntity]): Entity schemas from the E-R specification.

    Returns:
        GraphEntity: Matching entity schema or an empty `GraphEntity` if none.
    """
    node_type_formatted = node_type.replace("_", " ").lower()
    
    # Check if node type matches the names of any entity schemas
    matching_node_entity = next((entity_spec for entity_spec in node_schemas 
                                if (entity_spec.name.replace("_", " ").lower() == node_type_formatted)),
                              None)
    if matching_node_entity: 
        return matching_node_entity
    
    # Check if node type matching an example in an entity schema
    for entity_spec in node_schemas:
        matches_type_name = entity_spec.name.lower() == node_type_formatted
        matches_type_examples = any(node_type_formatted in example.lower() for example in entity_spec.examples)
        
        if (matches_type_name or matches_type_examples): 
            return entity_spec

    # Return an empty GraphEntity if node_type does not match any existing entity specifications
    return GraphEntity(node_type, "", [], "", {})


# ---- Schema ----    
def list_entity_types_and_attributes(graph, entity_specifications: List[GraphEntity], kg_rep: str) -> str:
    """
    Return entity types present in the KG and their associated schemas.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.
        entity_specifications (List[GraphEntity]): KG entity specifications.
        kg_rep (str): Output format strategy: `list` or `json`.

    Returns:
        str: Entities and schemas formatted according to `kg_rep`.
    """
    # Get all node types present in the graph + associated schemas
    node_types: List[str] = get_node_types_in_graph(graph)
    node_type_schemas: List[GraphEntity] = [get_schema_of_node_type(node_type, entity_specifications) for node_type in node_types]
    
    # Create return string
    if kg_rep == "list":
        node_type_schemas_str = f"{'\n'.join(map(lambda x: f'- {x.__repr__()}', set(node_type_schemas)))}"
    elif kg_rep == "json":
        node_type_schemas_str = f"```json\n{json.dumps({'entities': [
            {k: v for k, v in asdict(x).items() if v not in (None, "", {})}
            for x in node_type_schemas
        ]}, indent=None)}\n```"
    return f"{node_type_schemas_str}"

def list_relationship_schema(graph, entity_specifications: List[GraphEntity], relationship_specifications: List[GraphRelationship], kg_rep: str) -> str:
    """
    Return relationship schemas present in the KG filtered by existing nodes/edges.

    Applies targeted exclusions for overlapping cache-host relationships.

    Args:
        graph (nx.MultiDiGraph): Knowledge graph.
        entity_specifications (List[GraphEntity]): KG entity specifications.
        relationship_specifications (List[GraphRelationship]): KG relationship specifications.
        kg_rep (str): Output format strategy: `list` or `json`.

    Returns:
        str: Relationships formatted according to `kg_rep`.
    """
    def should_exclude_relationship(relationship: GraphRelationship, node_types_in_graph) -> bool:
        if relationship.source.lower() == "cache" and \
            relationship.target.lower() == "host" and \
                relationship.label == "hosted_on" and \
                    "Cache" in node_types_in_graph and\
                        "Cache_Instance" in node_types_in_graph:
                            return True
        elif relationship.source.lower() == "host" and \
            relationship.target.lower() == "cache" and \
                relationship.label == "hosts" and \
                    "Cache" in node_types_in_graph and\
                        "Cache_Instance" in node_types_in_graph:
                            return True
        else:
            return False
        
    node_types: List[str] = get_node_types_in_graph(graph)
    node_type_schemas: List[GraphEntity] = [get_schema_of_node_type(node_type, entity_specifications) for node_type in node_types]
    node_type_schema_names = {schema.name for schema in node_type_schemas}
    # Add examples from each schema to the set
    for schema in node_type_schemas:
        node_type_schema_names.update(schema.examples)
    edge_types: List[str] = get_edge_types_in_graph(graph)
    
    relationships_in_graph: List[GraphRelationship] = []
    for relationship in relationship_specifications:
        if (any(relationship.source.lower() in name.lower() for name in node_type_schema_names)) and \
           (any(relationship.target.lower() in name.lower() for name in node_type_schema_names)) and \
           (relationship.label in edge_types):
               if not should_exclude_relationship(relationship, node_types):
                   relationships_in_graph.append(relationship)
    
    if kg_rep == "list":
        relationships_in_graph_str = f"{'\n'.join(sorted(map(lambda x: f'- {x.__repr__()}', set(relationships_in_graph))))}"
    elif kg_rep == "json":
        relationships_in_graph_str = f"```json\n{json.dumps({'relationships': [
            {k: v for k, v in asdict(x).items() if v not in (None, "")}
            for x in relationships_in_graph
        ]}, indent=None)}\n```"
        
    return relationships_in_graph_str

# ---------------------------------------- GRAPH TOOLS ----------------------------------------    
# ---- Data Characteristics ----    
def check_node_existence(node: str, graph) -> bool:
    """Checks whether a node exists in the knowledge graph.

    Args:
        node (str): the identifier of the node.
        reasoning (str): the reasoning for calling this tool.

    Returns:
        bool: True if node exists in the graph; False otherwise.
    """    
    return node in graph

def get_node_attributes(node: str, graph) -> str:
    """Retrieves the attributes (e.g., type, anomaly alerts) of the given node in the knowledge graph.

    Args:
        node (str): the identifier of the node.
        reasoning (str): the reasoning for calling this tool.
    """
    if node not in graph:
        return f"Node `{node}` not found in the graph."

    attributes = graph.nodes[node]
    # attributes_str = ", ".join(map(lambda item: f"{item[0]}: {item[1]}", attributes).items())        
    attributes_str = format_attributes(attributes)       
    return f"The attributes of node `{node}` are:\n{attributes_str}."
    
def get_all_instances_of_entity_type(entity_type: str, graph) -> str:
    """Retrieves all the instances of a given entity type in the knowledge graph.

    Args:
        entity_type: the entity type for which to retrieve all instances
        reasoning (str): the reasoning for calling this tool and how its response will help to answer the original query.

    Returns:
        str: A string describing the node instances of the given entity type.
    """
    node_instances = []
    for (node, nodedata) in graph.nodes.items():
        if nodedata['type'] == entity_type:
            node_instances.append(node)
    
    if len(node_instances) == 0:
        return f"No nodes of entity type `{entity_type}` are present in the graph"
    
    return f"The nodes of entity type `{entity_type}` are: {', '.join(node_instances)}"

def get_edge_attributes(node1, node2, graph):
    """Retrieves the attributes (e.g., trace-level anomaly alerts) of the edge(s) between two nodes in the knowledge graph.

    Args:
        node1 (str): a node identifier.
        node2 (str): another node identifier.
    """
    def find_related_nodes(node):
        parent = children = None
        for _, n, _, data in graph.edges(node, keys=True, data=True):
            label = data.get('label')
            if label == "instance_of":
                parent = n
                break
            elif label == "has_instance":
                if not children:
                    children = []
                children.append(n)
        return parent, children
    
    def try_edge_pairs(sources, targets):
        for source in sources:
            for target in targets:
                if source and target and graph.has_edge(source, target):
                    return get_edge_attributes_helper(source, target, graph)
        return None
    
    def flatten(items):
        for x in items:
            if isinstance(x, list):
                yield from x
            else:
                yield x
    
    not_in_graph = ""
    if node1 not in graph:
        not_in_graph += f"Node `{node1}` not found in the graph. "
    if node2 not in graph:
        not_in_graph += f"Node `{node2}` not found in the graph. "
    if not_in_graph:
        return not_in_graph    
        
    # Find parent and child nodes
    node1_parent, node1_children = find_related_nodes(node1)
    node2_parent, node2_children = find_related_nodes(node2)
    
    # Build node variants (sum is to flatt to 1D list)
    node1_variants = list(flatten([node1, node1_parent or [], node1_children or []]))
    node2_variants = list(flatten([node2, node2_parent or [], node2_children or []]))
    
    forward_result = try_edge_pairs(node1_variants, node2_variants) or f"No edge found from `{node1_parent or node1}` to `{node2_parent or node2}`"
    backward_result = try_edge_pairs(node2_variants, node1_variants) or f"No edge found from `{node2_parent or node2}` to `{node1_parent or node1}`."   
    
    instances_str = "(across their instances) " if any([node1_parent, node2_parent, node1_children, node2_children]) else ""
    primary_node1 = node1_parent or node1
    primary_node2 = node2_parent or node2
    return_str = f"The attributes of the edge(s) between `{primary_node1}` and `{primary_node2}` {instances_str}are:\n"\
                 f"__Direction: `{primary_node1} --> {primary_node2}`__\n{forward_result}\n\n"\
                 f"__Direction: `{primary_node2} --> {primary_node1}`__\n{backward_result}"
    
    return return_str

# ---- Traversal ----  
def get_node_neighborhood(node: str, graph, strategy, r: int = 3) -> str:
    """
    Retrieves the r-hop neighborhood of a given node in the knowledge graph.
    The r-hop neighborhood consists of all nodes that are reachable from the given node within at most r hops.

    Args:
        node (str): The identifier of the node for which the r-hop neighborhood is to be computed.
        reasoning (str): the reasoning for calling this tool.
        r (int, optional): The maximum number of hops allowed to reach neighboring nodes. Defaults to 3.
    Returns:
        str: A string describing the set of nodes and edges that belong to the r-hop neighborhood of the given node.
    """    
    if node not in graph:
        return f"Node `{node}` not found in the graph."
    
    # Ensure r is an int
    r = int(r)
    
    # Perform BFS for downstream and upstream edges
    downstream_edges = list(nx.bfs_edges(graph, node, reverse=False, depth_limit=r))
    upstream_edges = list(nx.bfs_edges(graph, node, reverse=True, depth_limit=r))
    
    # Collect all visited nodes
    visited_nodes = {node}
    visited_nodes.update(u for u, v in downstream_edges)
    visited_nodes.update(v for u, v in downstream_edges)
    visited_nodes.update(u for u, v in upstream_edges)
    visited_nodes.update(v for u, v in upstream_edges)
    
    # Combine edges from both directions
    all_edges = downstream_edges + [(v, u) for u, v in upstream_edges]  # Reverse upstream edges to match the format
    
    if not visited_nodes or not all_edges:
        return f"No neighbors found within depth {r} of `{node}`."
    
    # Convert edges to labeled string
    if strategy == 'list':
        edges_str = convert_edges_to_list_string(graph, all_edges, is_reverse=False)
        nodes_str = convert_nodes_to_list_string(visited_nodes, with_type=False)
    elif strategy == 'json':
        edges_str = convert_edges_to_json_string(graph, all_edges, is_reverse=False)
        nodes_str = convert_nodes_to_json_string(visited_nodes, with_type=False)
        
    
    return f"r-hop neighborhood of `{node}` up to depth {r}:\nNodes: {nodes_str}.\nEdges:\n{edges_str}."

def get_all_simple_paths(source: str, target: str, graph) -> str:
    """
    Finds all simple paths from a source node to a target node in the knowledge graph.

    Args:
        source (str): The identifier of the source node.
        target (str): The identifier of the target node.
        reasoning (str): the reasoning for calling this tool.

    Returns:
        str: A string describing the simple paths between the source and target nodes.
    """
    try:
        if source not in graph and target not in graph:
            return f"Both the source and target nodes ({source}, {target}) are not found in the graph."
        if source not in graph:
            return f"The source node {source} is not found in the graph."
        if target not in graph:
            return f"The target node {target} is not found in the graph."
        
        # paths = list(nx.all_simple_paths(graph, source=source, target=target))
        # paths_str = "; ".join([f"{' -> '.join(path)}" for path in paths])
        
        paths = list(nx.all_simple_edge_paths(graph, source, target, cutoff=6))
        paths = sorted(paths, key=len)
        path_str_list = []
        for path in paths:
            labeled_edges = ""
            for i, edge in enumerate(path):
                u = edge[0]
                v = edge[1]
                edge_data = graph.get_edge_data(u, v)
                
                # If there are multiple edges between nodes, retrieve both edge labels
                if graph.is_multigraph():
                    labels = [f"{data.get('label', 'No label')}" for key, data in edge_data.items()]
                    if i == 0:
                        labeled_edges += (f"{u} --({', '.join(labels)})--> {v}")
                    else:
                        labeled_edges += (f" --({', '.join(labels)})--> {v}")
                else:
                    label = edge_data.get('label', 'No label')
                    if i == 0:
                        labeled_edges += (f"{u} --({label})--> {v}")
                    else:
                        labeled_edges += (f" --({label})--> {v}")
            path_str_list.append(labeled_edges)
        paths_str = "\n".join([f"- {path_str}" for path_str in path_str_list])
        if paths_str == "":
            paths_str = "None"
        return f"All simple paths from {source} to {target}:\n{paths_str}"
    except Exception as e:
        return f"Error: {e}"


# ---- ALL TOOLS ----  

def get_tools(kg, kg_representation) -> List[StructuredTool]:
    """
    Return the list of structured tools for querying a knowledge graph.

    Args:
        kg (nx.MultiDiGraph): Knowledge graph the tools operate on.
        kg_representation (str): Representation strategy: `list` or `json`.
    
    Returns:
        List[StructuredTool]: Structured tools bound to the given KG.
    """
    tools = [
        # Data Characteristics
        StructuredTool.from_function(
            func = lambda node, reasoning: check_node_existence(node, graph=kg),
            name = check_node_existence.__name__,
            description = check_node_existence.__doc__
        ),
        StructuredTool.from_function(
            func = lambda node, reasoning: get_node_attributes(node, graph=kg),
            name = get_node_attributes.__name__,
            description = get_node_attributes.__doc__
        ),
        StructuredTool.from_function(
            func = lambda node1, node2, reasoning: get_edge_attributes(node1, node2, graph=kg),
            name = get_edge_attributes.__name__,
            description = get_edge_attributes.__doc__
        ),
        StructuredTool.from_function(
            func = lambda type, reasoning: get_all_instances_of_entity_type(type, graph=kg),
            name = get_all_instances_of_entity_type.__name__,
            description = get_all_instances_of_entity_type.__doc__
        ),
        # Traversal
        StructuredTool.from_function(
            func = lambda node, reasoning, r = 3: get_node_neighborhood(node, r=r, graph=kg, strategy=kg_representation),
            name = get_node_neighborhood.__name__,
            description = get_node_neighborhood.__doc__
        ),
        StructuredTool.from_function(
            func = lambda source, target, reasoning: get_all_simple_paths(source, target, graph=kg),
            name = get_all_simple_paths.__name__,
            description = get_all_simple_paths.__doc__
        ),
    ]
    return tools
