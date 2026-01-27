"""
gradio-messages-agent.py

Gradio UI to run RCA fault scenarios with LangGraph agents in real-time.

It loads fault alert data and the system knowledge graph (KG) for the selected
system. It lets you select the agent type, LLM model, fault id, and alert/KG
representations, then streams the agent's messages and any structured final
response to the UI for inspection.

Examples:
    python code/gradio-apps/gradio-messages-agent.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import traceback
import gradio as gr
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.messages.base import get_msg_title_repr
from agents.graphs import *
from agents.tools import *
from agents.structured_output import *
from rca import process_fault_data, get_all_data_for_prompt_context
from utils.io import load_config, load_fault_data, load_prompt_templates
from utils.llm import load_model_ollama
from utils.kg import load_knowledge_graph_with_alert_data, load_graph_entity_relationship_specifications

MAX_STEPS = 50

code_path_prefix = "code/"
data_path_prefix = "data/"
KG_PATH = data_path_prefix + "system-knowledge-graphs/original/{system}/"
model_name = "llama3.3"
schema = RootCausesResponse

config = load_config(code_path_prefix + 'config/rca-config.yaml')

# templates for all settings
templates = load_prompt_templates(code_path_prefix + f"config/prompts/effect-to-cause/", None)
entity_specifications, relationship_specifications = load_graph_entity_relationship_specifications(data_path_prefix + "entity-relationships/entity-relationships-specification.json")

def load_system_data(system, modality_to_exclude):
    fault_data = load_fault_data(
        data_path_prefix + f"fault-alerts/{system}/log/logs.json",
        data_path_prefix + f"fault-alerts/{system}/trace/traces.json",
        data_path_prefix + f"fault-alerts/{system}/metric/metrics.json",
        modality_to_exclude
    )
    fault_data = process_fault_data(fault_data, config[system]['alert_timezone'])
    
    kg_path = KG_PATH.format(system=system)
    kg_file = kg_path + f'{system}-KG.json'
    
    return kg_file, fault_data

agent_type_to_workflow_lookup = {
    "simple": initialize_simple_graph,
    "simple with structured output": initialize_simple_graph_structured_output,
    "simple with two-step structured output": initialize_simple_graph_structured_output_two_step,
    "react": initialize_react_graph,
    "react with tool retry": initialize_react_graph_with_tool_retry,
    "react with structured output": initialize_react_graph_with_structured_output,
    "react with tool retry and structured output": initialize_react_graph_with_tool_retry_and_structured_output,
    "plan-and-execute": initialize_plan_execute_react_shared_state
}

def initialize_graph_master(model: BaseChatModel, tools, agent_type):
    model_with_tools = model.bind_tools(tools)
    model_with_structured_output = model.with_structured_output(schema=schema)
    graph = None
    setting_templates = None
    match agent_type:
        case "simple":
            setting_name = "model_only_full"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_simple_graph(model)
        case "simple with structured output":
            setting_name = "model_only_full"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_simple_graph_structured_output(model_with_structured_output)
        case "simple with two-step structured output":
            setting_name = "model_only_full"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_simple_graph_structured_output_two_step(model, model_with_structured_output, setting_templates)
        case "react":
            setting_name = "react"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_react_graph(model_with_tools, tools)
        case "react with tool retry": 
            setting_name = "react"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_react_graph_with_tool_retry(model_with_tools, tools)
        case "react with structured output":
            setting_name = "react"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_react_graph_with_structured_output(model_with_tools, tools, model_with_structured_output, setting_templates)
        case "react with tool retry and structured output":
            setting_name = "react"
            setting_templates = {**templates, 
                                "rca": templates["rca"].get(setting_name, {})}
            graph = initialize_react_graph_with_tool_retry_and_structured_output(model_with_tools, tools, model_with_structured_output, setting_templates)
        case "plan-and-execute":
            setting_name = "plan_execute_react"
            setting_templates = templates
            graph = initialize_plan_execute_react_shared_state(model, model_with_tools, tools, model_with_structured_output, setting_templates)
    return graph, setting_templates

def fix_markdown_formatting(message: BaseMessage):
    if message.type == 'ai' and message.tool_calls:
        title = get_msg_title_repr(message.type.title() + " Message")
        message_str = message.pretty_repr()
        content = message_str.replace(title, "")
        # Wrap tool call info in ```text``` block.
        text = f"{title}\n\n```text\n{content}\n```"

    else:
        text = message.pretty_repr()
        # Replace all single newlines (that are not part of a list item) with soft breaks (\\\n)
        text = re.sub(r'(?<!\n)\n(?!\n|\s*([-*+]|[0-9]+\.)\s)', '\\\n', text)
    return text


def run_agent(system, model_name, agent_type, fault_id, alert_rep, kg_rep, modalities):
    """
    Orchestrates the Plan-and-Execute ReAct LangGraph agent with live UI updates.
    """
    if modalities == "all":
        modality_to_exclude = None
    else:
        modality_to_exclude = modalities.split("exclude-")[1]
        
    kg_file, fault_data = load_system_data(system, modality_to_exclude)
    kg = load_knowledge_graph_with_alert_data(kg_file, fault_id, fault_data)
    alerts_for_fault = fault_data[fault_id]
    tools = get_tools(kg, kg_rep)
    model = load_model_ollama(model_name,-2)
    graph, setting_templates = initialize_graph_master(model, tools, agent_type)
    
    all_data = get_all_data_for_prompt_context(system, alerts_for_fault, kg, entity_specifications, relationship_specifications, alert_rep, kg_rep)
    
    if agent_type == "plan-and-execute":
        messages = [
            SystemMessage(content = setting_templates['plan_execute']['input'].format(**all_data))
        ]
    else:
        messages = [
            SystemMessage(content = setting_templates['rca']['system'].format(**all_data)),
            HumanMessage(content = setting_templates['rca']['human'].format(**all_data))
        ]
    
    output_text = "\n\n".join([message.pretty_repr() for message in messages])
    yield gr.Markdown(value=output_text)

    # To stream events and process them in real-time: replace following line with: `for s in graph.stream(...)`
    try:
        s = graph.invoke({"messages": messages}, config={"recursion_limit": 50, "configurable": {"thread_id": "123abc"}}, stream_mode = "values")
        # Replace all single newlines not part of a double newline ONLY if the next line does not start with a list item (-, *, or numbered list like 1.)
        output_text = "\n\n".join([fix_markdown_formatting(message) 
                                       for message in s['messages'] ])
        print(output_text)
        if "final_response" in s.keys():
            if s['final_response']:
                final_response_str = json.dumps(s['final_response'], indent=2)
                output_text += f"\n\n### Final response:\n\n```json\n{final_response_str}\n```"
            else:
                output_text += f"\n\n### Final response:\n\nNone"
        yield gr.Markdown(value=output_text)
    except Exception as e:
        state = graph.get_state({"thread_id": "123abc"})
        print(state)
        output_text += f"{traceback.print_exception(e)}\n\n"
        yield gr.Markdown(value=output_text)
        
    output_text += "\n\n### Execution Complete!"
    yield gr.Markdown(value=output_text)


# Build Gradio UI
def build_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("### Agent with Messages")
        
        with gr.Row():
            system_dropdown = gr.Dropdown(
                label="System",
                choices = ["MicroSS", "Online-Boutique-cloudbed-1", "Online-Boutique-cloudbed-2"],
                interactive=True
            )
            agent_type_dropdown = gr.Dropdown(
                label="Type of agent",
                choices = list(agent_type_to_workflow_lookup.keys()),
                interactive=True
            )
            alert_rep_dropdown = gr.Dropdown(
                label="Alert representation",
                choices=['by-time', 'by-component'],
                interactive=True
            )
            kg_rep_dropdown = gr.Dropdown(
                label="KG representation",
                choices = ['list', 'json'],
                interactive=True
            )
            modalities_dropdown = gr.Dropdown(
                label="Alert modalities",
                choices = ['all', 'exclude-logs', 'exclude-traces', 'exclude-metrics']
            )
            model_dropdown = gr.Dropdown(
                label="LLM",
                choices= ["llama3.2:3b", "qwen3:4b", "qwen3:32b", "llama3.3", "command-r-plus"], 
                value="llama3.3",
                interactive=True
            )
            fault_id = gr.Dropdown(
                label="Fault id",
                choices = list(map(str, range(700))),
                value="0",
                interactive=True
            )
        
        with gr.Row():
            start_btn = gr.Button("Start Agent")
            reset_btn = gr.Button("Reset")    
        
        with gr.Row():
            output = gr.Markdown("Waiting for execution...")
        
        # Click event starts the agent
        start_btn.click(run_agent, inputs=[system_dropdown, model_dropdown, agent_type_dropdown, fault_id, alert_rep_dropdown, kg_rep_dropdown, modalities_dropdown], outputs=output)
        
        # Reset button hides all accordions and clears Markdown content
        reset_btn.click(lambda: "Waiting for execution...",
                        outputs= output)

    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(share=True)
