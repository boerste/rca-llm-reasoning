import json
import networkx as nx
from typing import List, Any, Optional, Dict
from dataclasses import dataclass

from fault_scenarios import LogAlert, MetricAlert, TraceAlert
from utils.io import load_json

@dataclass(frozen=True)
class GraphEntity():
    """
    Represents an entity (node) TYPE in the knowledge graph specification.

    Attributes:
        name (str): The logical name or type of the entity.
        description (str): A description of the entity type's role or purpose.
        examples (List[str]): Examples of real-world instances of this entity type.
        attributes (str): Description of attributes associated with this entity type.
        other (Dict[str, Any]): Additional metadata.
    """
    name: str
    description: str
    examples: List[str]
    attributes: str
    other: Dict[str, Any]
    
    def __repr__(self):
        return f"{self.name}: {self.description} Examples: {', '.join(self.examples) if len(self.examples) else 'None'}. Attributes: {self.attributes if (self.attributes != '') else 'None'}."
        # "{json.dumps(self.other) if self.other else 'None'}"

    def __hash__(self):
        return hash((self.name, self.description, tuple(self.attributes), tuple(self.examples)))

    def __eq__(self, other):
        return (self.name, self.description, self.examples, self.attributes,) == \
               (other.name, other.description, other.examples, other.attributes)

@dataclass
class GraphRelationship:
    """
    Represents a relationship (edge) TYPE between two entities in the knowledge graph specification.

    Attributes:
        source (str): The source entity type.
        target (str): The target entity type.
        label (str): The relationship label (e.g., 'calls', 'hosted_on').
        description (str): A description of what the relationship implies.
    """
    source: str
    target: str
    label: str
    description: str
    
    def __repr__(self):
        repr = f"{self.source} --({self.label})--> {self.target}"
        if self.description:
            repr += f", description: {self.description}"
        return repr
    
    def __hash__(self):
        return hash((self.source, self.target, self.label, self.description))
    
    def __eq__(self, other):
        return (self.source, self.target, self.label, self.desription) == \
               (other.source, other.target, other.label, other.description)
    
def load_knowledge_graph(kg_file, graph_type = nx.MultiDiGraph, source_type="json", include_print = True ) -> nx.Graph:
    """
    Load knowledge graph from a JSON or JSON-LD file into a NetworkX MultiDiGraph.

    Args:
        kg_file (str): Path to the knowledge graph file.
        graph_type (class): The NetworkX graph class to use (default: nx.MultiDiGraph).
        source_type (str): The format of the source file, 'json' or 'jsonld' (default: 'json').
        include_print (bool): Whether to print loading status (default: True).

    Returns:
        nx.Graph: The loaded NetworkX graph.

    Raises:
        ValueError: If file is not found or has invalid format.
    """
    if include_print:
        print(f"Loading knowledge graph {kg_file} as {graph_type}...")
    if source_type == "json":
        try:
            with open(kg_file) as f:
                data = json.load(f)
                
            G = graph_type()
            for node in data.get('nodes', []):
                G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            for edge in data.get('edges', []):
                G.add_edge(edge['source'], edge['target'], label=edge['label'])
        except FileNotFoundError:
            raise ValueError(f"Knowledge graph file '{kg_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file '{kg_file}'.")
    
    if source_type == "jsonld":
        from rdflib import Graph
        # Load JSON-LD data into an RDF graph
        rdf_graph = Graph()
        rdf_graph.parse(kg_file, format="json-ld")

        G = graph_type()
        # Iterate through RDF triples to add edges to the NetworkX graph
        for subj, pred, obj in rdf_graph:
            G.add_edge(subj, obj, predicate=pred)

    return G

def load_graph_entity_relationship_specifications(path: str) -> tuple[List[GraphEntity], List[GraphRelationship]]:
    """
    Loads the specification of the entities and relationships from file.

    Args:
        path (str): Path to the JSON specification file.

    Returns:
        tuple[List[GraphEntity], List[GraphRelationship]]: A tuple containing a list of graph entities and a list of graph relationships.
    """
    json_data = load_json(path)
    entities = []
    relationships = []
    
    # Parse nodes (entities)
    for entity in json_data.get("entities", []):
        name = entity["id"]
        description = entity.get("definition", "")
        attributes = entity.get("attributes", "")
        other = {k: v for k, v in entity.items() if k not in {"id", "definition", "examples", "note", "attributes"}}
        examples = entity.get("examples", "").split(", ") if "examples" in entity else []
        
        entities.append(GraphEntity(name, description, examples, attributes, other))
    
    # Parse edges (relationships)
    for relationship in json_data.get("relationships", []):
        relationships.append(
            GraphRelationship(
                relationship["source"],
                relationship["target"],
                relationship["label"],
                relationship.get("description", "")
            )
        )
    
    return entities, relationships

def load_knowledge_graph_with_alert_data(kg_file: str, fault_id: str, fault_alerts: dict, graph_type = nx.MultiDiGraph, source_type="json", include_print = True):
    """
    Load knowledge graph from JSON or JSON-LD file into a NetworkX MuliDiGraph.
    Nodes additionally contain 'node status' -- loaded from the multi-modal telemetry alerts per fault id.
    
    Args:
        kg_file (str): Path to the knowledge graph file.
        fault_id (str): The ID of the fault scenario to enrich the graph with.
        fault_alerts (dict): Dictionary mapping fault IDs to their alerts.
        graph_type (class): The NetworkX graph class to use (default: nx.MultiDiGraph).
        source_type (str): The format of the source file (default: 'json').
        include_print (bool): Whether to print loading status (default: True).

    Returns:
        nx.Graph: The loaded NetworkX graph enriched with alert data on nodes and edges.
    """
    G = load_knowledge_graph(kg_file, graph_type, source_type, include_print)
    alerts_for_fault = fault_alerts.get(fault_id, {})
   
    # Add logs to node attributes
    log_alert: LogAlert
    for log_alert in alerts_for_fault['log_alerts']:
        node = log_alert.component
        if node not in G:
            print(f"Log processing: {node} not in graph")
            continue
        node_logs = G.nodes[node].get('log_alerts', [])
        G.nodes[node]['log_alerts'] = node_logs + [log_alert.format_for_kg()]
        
    # Add metrics to node attributes
    metric_alert: MetricAlert
    for metric_alert in alerts_for_fault['metric_alerts']:
        node = metric_alert.component
        if node not in G:
            print(f"Metric processing: {node} not in graph")
            continue
        node_metrics = G.nodes[node].get('metric_alerts', [])
        G.nodes[node]['metric_alerts'] = node_metrics + [metric_alert.format_for_kg()]

    # Add trace events to edge attributes
    trace_alert: TraceAlert
    for trace_alert in alerts_for_fault['trace_alerts']:
        if trace_alert.source not in G:
            print(f"Trace processing: {trace_alert.source} not in graph")
            continue
        if trace_alert.target not in G:
            print(f"Trace processing: {trace_alert.target} not in graph")
            continue
        source_parent = None
        for _, n, key, data in G.edges(trace_alert.source, keys=True, data=True):
            if data.get('label') == "instance_of":
                source_parent = n
                break
        target_parent = None
        for _, n, key, data in G.edges(trace_alert.target, keys=True, data=True):
            if data.get('label') == "instance_of":
                target_parent = n
                break 
        
        # TODO: eventually sort trace_alerts by trace_alert.type (i.e. ERROR first, then PD)
        edges = G.get_edge_data(source_parent, target_parent, default={})
        if edges == {}:
            print(f"[fault id: {fault_id}] No edge exists between {source_parent} and {target_parent}")
        for key, data in edges.items():
            if data.get('label') in {"control_flow", "data_flow"}:
                edge_traces = data.get('trace_alerts', [])
                G[source_parent][target_parent][key]['trace_alerts'] = edge_traces + [trace_alert.format_for_kg()]

    for u, v, key, data in G.edges(keys=True, data=True):
        # print(f"{u} -> {v} (key={key}): {data}")
        pass
    
    return G



## --------------------------- Graph-related utility functions ---------------------------
def graph_details_with_labels(G: nx.DiGraph):
    """
    Get graph details including node count, list of nodes, and edges with labels.

    Args:
        G (nx.DiGraph): The input graph.

    Returns:
        tuple: (number_of_nodes, list_of_nodes, edges_flattened_string)
    """
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    edges_with_labels = [(u, v, d.get('label')) for u, v, d in G.edges(data=True)]
    edges_flat = " ".join(f"({u}->{v}, {d})" for u, v, d in edges_with_labels)
    return num_nodes, nodes, edges_flat

def graph_details(G: nx.DiGraph):
    """
    Get graph details including node count, list of nodes, and edges without labels.

    Args:
        G (nx.DiGraph): The input graph.

    Returns:
        tuple: (number_of_nodes, list_of_nodes, edges_flattened_string)
    """
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    edges = G.edges()
    edges_flat = " ".join(f"({u}->{v})" for u, v in edges)
    return num_nodes, nodes, edges_flat
