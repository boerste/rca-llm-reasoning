"""
rca.py

Run RCA agent pipelines over fault scenarios using local/remote LLMs.

Purpose:
- Load knowledge graphs and fault alert telemetry
- Construct structured prompts per setting (model_only, react, plan_execute_react)
- Invoke LLM agents (optionally with tools) and collect structured outputs
- Write per-scenario results to JSONL for downstream evaluation

Requirements:
- Config YAML specifying systems, models, settings, prompts, and file paths
- Knowledge graph files and fault telemetry (logs/traces/metrics)
- Ollama or OpenRouter configured depending on the selected model

Key Options:
- Settings: `model_only`, `react`, `plan_execute_react`
- Tasks: `effect_to_cause`
- Representations: alerts (`by-time`, `by-component`), KG (`list`, `json`)
- Optional: modalities to exclude using `--exclude-modalities <MODALITY>` flag (MODALITY = `logs`, `metrics` or `traces`)

Examples:
    python code/rca.py --config code/config/rca-config.yaml --system-name MicroSS --model llama3.3 --settings "model_only react" --tasks cause_to_effect
    python code/rca.py --config code/config/rca-config.yaml --system-name Online-Boutique --kg-rep json --alert-rep by-component --exclude-modalities logs

"""

import os
import time
import argparse
import traceback
import logging
import uuid
import pandas as pd
from typing import List
from tqdm import tqdm
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import StateSnapshot
from agents.graphs import initialize_simple_graph_structured_output_two_step, initialize_react_graph_with_tool_retry_and_structured_output, initialize_plan_execute_react_shared_state
from agents.tools import get_tools, get_all_edges, list_entity_types_and_attributes, list_relationship_schema, get_all_nodes
from agents.structured_output import *
from fault_scenarios import process_fault_data, format_alerts_for_fault, is_alerts_non_empty_for_fault, get_possible_entity_types_for_root_cause_location, get_possible_fault_types
from utils.logging import setup_logger
from utils.io import *
from utils.kg import *
from utils.llm import ModelResult, load_model_ollama, load_model_openrouter, get_model_with_structured_output
from utils.common import get_kg_id_from_file_name

LABEL_CURRENT_SYSTEM = "current_system"

@dataclass
class SettingConfig:
    use_react: bool
    file_name: str
    class_name: str
    input_key: str
    state_key: str

    @staticmethod
    def from_dict(dict):
        """
        Create a SettingConfig instance from a dictionary.

        Args:
            dict (dict): Dictionary representing a setting configuration.

        Returns:
            SettingConfig: Instantiated configuration object for a setting.
        """
        return SettingConfig(
            use_react=dict["use_react"],
            file_name=dict["file_name"],
            class_name=dict["class_name"],
            input_key=dict["input_key"],
            state_key=dict["state_key"]
        )

def get_all_data_for_prompt_context(
    system: str,
    alerts_for_fault: dict,
    kg,
    entity_specifications,
    relationship_specifications,
    alert_representation: str = "by-time",
    kg_representation: str = "list",
    ):
    """
    Build the `all_data` context dictionary used for RCA prompt templates for a SINGLE fault.

    Collects and formats fault alerts, KG schema, tool names, and KG nodes/edges
    to supply placeholders for system/human prompts.

    Args:
        system (str): Name of the system in which the fault occurred.
        alerts_for_fault (dict): Alerts for the specific fault.
        kg: Knowledge graph instance.
        entity_specifications: KG entity specifications.
        relationship_specifications: KG relationship specifications.
        alert_representation (str): Alert representation style.
        kg_representation (str): KG representation style.

    Returns:
        dict: Aggregated data for template formatting.
    """
    # Format the alerts
    alerts_for_fault_formatted = format_alerts_for_fault(alerts_for_fault, alert_representation)

    # Retrieve additional information to insert into the rca templates
    entity_schema_str = list_entity_types_and_attributes(
        kg, entity_specifications, kg_representation)
    relationships_schema_str = list_relationship_schema(
        kg, entity_specifications, relationship_specifications, kg_representation)
    fault_types = get_possible_fault_types(system)
    root_cause_fault_entity_types = get_possible_entity_types_for_root_cause_location(system)
    
    tools = get_tools(kg, kg_representation)
    tools_by_name = ", ".join([f"`{tool.name}`" for tool in tools])
    
    nodes_str = get_all_nodes(kg, kg_representation)
    edges_str = get_all_edges(kg, kg_representation)
    
    alert_rep_str = alert_representation.replace("-", " ")

    additional_data_dict = {"alerts": alerts_for_fault_formatted,
                            "entity_schema": entity_schema_str,
                            "relationship_schema": relationships_schema_str,
                            "knowledge_graph_nodes": nodes_str,
                            "knowledge_graph_edges": edges_str,
                            "root_cause_fault_types": fault_types,
                            "root_cause_fault_entity_types": root_cause_fault_entity_types,
                            "tools_by_name": tools_by_name,
                            "alert_rep": alert_rep_str}

    # Merge dictionaries together so all necessary keys are available for string formatting
    all_data = {**additional_data_dict}
    return all_data
    
class BasePipeline:
    """
    Base class for an RCA execution pipeline.

    Encapsulates common logic for initializing agent graphs, constructing prompts,
    invoking the agent graph, and packaging results.
    """
    def __init__(
        self,
        config,
        config_all,
        model,
        structured_output_class,
        kg,
        kg_representation,
        prompt_templates
    ):
        self.config = SettingConfig.from_dict(config) # config specific to the setting
        self.config_all = config_all # all config
        self.model = model # model not yet binded to tools or with structured output
        self.structured_output = structured_output_class
        self.kg_rep = kg_representation
        self.initialize_agent_graph(model, structured_output_class, kg, kg_representation, prompt_templates)
        
    def initialize_agent_graph(self, model: BaseChatModel, structured_output_class, kg, kg_representation, prompt_templates):
        """
        Initialize the agent graph for a specific pipeline setting.

        Subclasses must override this to bind tools and/or structured output
        and create the appropriate LangGraph graph.
        """
        logging.warning("Agent graph not initialized! Should be implemented by subclass.")
        return None
        
    def construct_prompt(
        self,
        system: str,
        alerts_for_fault: dict,
        template: dict,
        kg,
        entity_specifications,
        relationship_specifications,
        alert_representation: str,
        kg_representation: str
    ):
        """
        Build RCA messages (system + human) for the given setting for a SINGLE fault.

        Args:
            system (str): Name of the system in which the fault occurred.
            alerts_for_fault (dict): Alerts dict for the fault.
            template (dict): Setting-specific templates containing `rca` keys.
            kg: Knowledge graph instance.
            entity_specifications: KG entity specifications.
            relationship_specifications: KG relationship specifications.
            alert_representation (str): Alert representation style.
            kg_representation (str): KG representation style.

        Returns:
            List[BaseMessage]: Ordered RCA messages for the agent.
        """
        # NOTE: template should be the setting-specific template (with 'rca' key)
        rca_template = template["rca"]
        all_data = get_all_data_for_prompt_context(
            system,
            alerts_for_fault,
            kg,
            entity_specifications,
            relationship_specifications,
            alert_representation,
            kg_representation
        )

        rca_messages = [
            SystemMessage(content=rca_template["system"].format(**all_data)),
            HumanMessage(content=rca_template["human"].format(**all_data))
        ]
        return rca_messages
    
    def model_invoke(self, input):
        """
        Invoke the agent graph with the given input and return outputs.

        Creates a unique `thread_id`, runs the graph with recursion limits,
        and cleans up state afterwards. Returns state values or error info.

        Args:
            input (Any): Graph input (messages or setting-specific payload).

        Returns:
            dict: Graph response including state values or error details.
        """
        key = self.config.input_key
        thread_id = str(uuid.uuid4())
        config = {
            "recursion_limit": self.config_all['recursion_limit'],
            "configurable": {
                "thread_id": thread_id
            }
        }
        try:
            answer = self.agent_graph.invoke(
                {key: input}, 
                config
            )
            return answer
        except Exception as e:
            logging.error("Error during model invoke.", exc_info=True)
            # Retrieve partial state to better understand error
            state: StateSnapshot = self.agent_graph.get_state(config)
            return {key: input, "error": traceback.format_exception(e), **state.values}
        finally:
            # always delete the state whether success or failure
            self.agent_graph.checkpointer.delete_thread(thread_id=thread_id)
        
    def evaluate(
        self,
        system: str,
        alerts_for_fault: dict,
        fault_id: str,
        kg,
        kg_id: str,
        model: BaseChatModel,
        prompt_template: dict,
        alert_representation: str,
        kg_representation: str,
        entity_specifications,
        relationship_specifications
    ) -> ModelResult:
        """
        Execute a SINGLE fault scenario and return a `ModelResult`.

        Skips execution if no alerts are present. Otherwise constructs the prompt,
        invokes the agent graph, and packages the result.

        Args:
            system (str): Name of the system in which the fault occurred.
            alerts_for_fault (dict): Alerts for the fault.
            fault_id (str): Fault identifier.
            kg: Knowledge graph instance.
            kg_id (str): Unique KG identifier.
            model (BaseChatModel): Underlying LLM chat model instance to invoke.
            prompt_template (dict): Templates for the setting.
            alert_representation (str): Alert representation style.
            kg_representation (str): KG representation style.
            entity_specifications: KG entity specifications.
            relationship_specifications: KG relationship specifications.

        Returns:
            ModelResult: Structured result with messages, final_response, timing, and errors.
        """
        if not is_alerts_non_empty_for_fault(alerts_for_fault):
            # Skip evaluation of there are no alerts for this fault.
            return ModelResult(
                kg_id,
                fault_id,
                None,
                None,
                0,
                "Not executed. No alerts detected for fault.",
                None
            )
            
        start_time = time.time()
        input = self.construct_prompt(
            system,
            alerts_for_fault,
            prompt_template,
            kg,
            entity_specifications,
            relationship_specifications,
            alert_representation,
            kg_representation
        )
        response = self.model_invoke(input)
        end_time = time.time()

        total_time = end_time - start_time
        result = ModelResult(
            kg_id,
            fault_id,
            response.get(self.config.state_key, None),
            response.get('final_response', None),
            total_time,
            response.get('error', None),
            response.get('past_steps', None) # only for plan-execute
        )
        
        return result

class SimpleAgentPipeline(BasePipeline):
    """Pipeline that uses model-only with structured output (two-step)."""
    def initialize_agent_graph(self, model: BaseChatModel, structured_output_class, kg, kg_representation, prompt_templates):
        model_with_structured_output = get_model_with_structured_output(model, structured_output_class)
        self.agent_graph = initialize_simple_graph_structured_output_two_step(model, model_with_structured_output, prompt_templates)

class ReactAgentPipeline(BasePipeline):
    """Pipeline that binds tools (ReAct) and enforces structured outputs."""
    def initialize_agent_graph(self, model: BaseChatModel, structured_output_class, kg, kg_representation, prompt_templates):
        tools = get_tools(kg, kg_representation)
        model_with_tools = model.bind_tools(tools)
        model_with_structured_output = get_model_with_structured_output(model, structured_output_class)
        self.agent_graph = initialize_react_graph_with_tool_retry_and_structured_output(model_with_tools, tools, model_with_structured_output, prompt_templates)

class PlanExecuteAgentPipeline(BasePipeline):
    """Pipeline with shared-state plan-execute ReAct and structured outputs."""
    def initialize_agent_graph(self, model: BaseChatModel, structured_output_class, kg, kg_representation, prompt_templates):
        tools = get_tools(kg, kg_representation)
        model_with_tools = model.bind_tools(tools)
        model_with_structured_output = get_model_with_structured_output(model, structured_output_class)
        self.agent_graph = initialize_plan_execute_react_shared_state(model, model_with_tools, tools, model_with_structured_output, prompt_templates)
    
    def construct_prompt(self, system: str, alerts_for_fault: dict, template: dict, kg, entity_specifications, relationship_specifications, alert_representation: str, kg_representation: str):
        """
        Build prompt messages for the plan-execute setting for a SINGLE fault.

        Args:
            system (str): Name of the system in which the fault occurred..
            alerts_for_fault (dict): Alerts dict for the fault.
            template (dict): Templates containing `plan_execute` inputs.
            kg: Knowledge graph instance.
            entity_specifications: KG entity specifications.
            relationship_specifications: KG relationship specifications.
            alert_representation (str): Alert representation style.
            kg_representation (str): KG representation style.

        Returns:
            List[BaseMessage]: Messages list containing the plan input.
        """
        pe_template = template["plan_execute"]
        all_data = get_all_data_for_prompt_context(system, alerts_for_fault, kg, entity_specifications, relationship_specifications, alert_representation, kg_representation)

        messages = [
            SystemMessage(content=pe_template['input'].format(**all_data))
        ]
        return messages

class ModelOnlySetting(SimpleAgentPipeline):
    pass
        
class ReactSetting(ReactAgentPipeline):
    pass

class PlanExecuteReactSetting(PlanExecuteAgentPipeline):
    pass

def run_test(
    config: dict,
    setting_name: str,
    setting_class,
    kg_file: str,
    fault_data: dict,
    fault_gt: pd.DataFrame,
    structured_output_class,
    alert_representation: str,
    kg_representation: str,
    modality_to_exclude_desc: str,
    model: BaseChatModel,
    prompt_templates,
    entity_specifications: List[GraphEntity],
    relationship_specifications: List[GraphRelationship],
    write_responses: bool,
    results_dir: str,
    select_samples_df: pd.DataFrame | None,
    start_idx: int = 0,
    stop_idx: int | None = None
):
    """
    Run RCA inference for a single setting across a slice of fault scenarios.

    Iterates rows in ground-truth dataframe, loads per-fault KG + alerts, executes
    the pipeline, and writes results periodically.

    Args:
        config (dict): Global configuration dictionary.
        setting_name (str): Setting identifier (e.g., `model_only`).
        setting_class: Pipeline class to instantiate for the setting.
        kg_file (str): Path to the KG file used for this batch.
        fault_data (dict): Alerts dataset for logs/traces/metrics.
        fault_gt (pd.DataFrame): Ground-truth dataframe of injected faults.
        structured_output_class: TypedDict/class for structured outputs.
        alert_representation (str): Alert representation style.
        kg_representation (str): KG representation style.
        modality_to_exclude_desc (str): Modality exclusion description.
        model (BaseChatModel): Chat model instance to invoke.
        prompt_templates (dict): Templates used by the setting.
        entity_specifications (List[GraphEntity]): KG entity specifications.
        relationship_specifications (List[GraphRelationship]): KG relationship specifications.
        write_responses (bool): Whether to write results to file.
        results_dir (str): Directory where results will be written.
        select_samples_df (pd.DataFrame | None): Optional selective rerun filter.
        start_idx (int): Start index within `fault_gt`.
        stop_idx (int | None): Stop index (exclusive).

    Returns:
        None
    """
    if stop_idx is None:
        stop_idx = len(fault_gt) # run to the end
    num_scenarios_to_run = (stop_idx - start_idx)
    logging.info(f"Executing {num_scenarios_to_run} scenarios from index {start_idx} to {stop_idx}.")
    
    results_file = os.path.join(results_dir, config[setting_name]['file_name'])
    results: list[ModelResult] = []
    
    pbar = tqdm(fault_gt[start_idx:stop_idx].iterrows(), desc="Evaluating scenarios", total=num_scenarios_to_run)
    # Each fault-alerts set will have a unique G with alerts and setting instance
    for i, row in pbar:
        fault_id: str = str(row['fault_id'])

        meter = pbar.format_meter(pbar.n, pbar.total, pbar.format_dict["elapsed"])
        logging.info(
            f"fault id: {fault_id} - "
            f"{meter}"
        )
        
        if select_samples_df is not None:
            is_sample_selected = select_samples_df[
                (select_samples_df['system']==config[LABEL_CURRENT_SYSTEM]) &
                (select_samples_df['model']==model.model) &
                (select_samples_df['setting']==setting_name) &
                (select_samples_df['alerts_rep']==alert_representation) &
                (select_samples_df['kg_rep']==kg_representation) &
                (select_samples_df['fault_id'].astype(str)==fault_id) &
                (select_samples_df['modalities']==modality_to_exclude_desc)
            ]
            should_rerun = bool(len(is_sample_selected))
            if not should_rerun:
                continue

        G = load_knowledge_graph_with_alert_data(kg_file,
                                                fault_id,
                                                fault_data,
                                                include_print = False)
        setting_instance: BasePipeline = setting_class(config[setting_name],
                                                       config,
                                                       model,
                                                       structured_output_class,
                                                       G,
                                                       kg_representation,
                                                       prompt_templates)
        alerts_for_fault = fault_data.get(fault_id, {})
        result: ModelResult = setting_instance.evaluate(config[LABEL_CURRENT_SYSTEM],
                                                        alerts_for_fault,
                                                        fault_id,
                                                        G,
                                                        get_kg_id_from_file_name(kg_file),
                                                        model,
                                                        prompt_templates,
                                                        alert_representation,
                                                        kg_representation,
                                                        entity_specifications,
                                                        relationship_specifications)
        
        results.append(result)
        
        # Save results according to write_results_freq
        if write_responses and (i % config['write_results_freq'] == 0):
            logging.info(f"Writing {len(results)} to file {results_file}")
            for result in results:
                write_to_file(result, results_file, include_print=False)
            results = []
    
    # Save the remaining results
    for result in results:
        write_to_file(result, results_file, include_print=False)

def run_tests(config: dict,
              kg_files: List[str],
              fault_data: dict,
              fault_gt: pd.DataFrame,
              model_name: str,
              settings: List[str],
              tasks: List[str],
              alert_reps: List[str],
              kg_reps: List[str],
              modality_to_exclude_desc: str,
              write_responses = True,
              results_dir = "",
              select_samples_df: pd.DataFrame | None = None,
              start_idx: int = 0,
              stop_idx: int | None = None,
              context_length: int = 12288,
              reasoning_effort: str = None,
              reasoning_max_tokens: int = None):
    """
    Run RCA tests across configurations, KGs, and scenarios for one system and LLM model.

    Orchestrates prompt template loading, model initialization (Ollama or OpenRouter),
    and iterates over representations/settings to execute `run_test` for each KG.

    Args:
        config (dict): Global configuration dictionary.
        kg_files (List[str]): Knowledge graph file paths for the system.
        fault_data (dict): Alerts indexed by fault id: `logs`, `traces`, `metrics`.
        fault_gt (pd.DataFrame): Ground-truth faults dataframe.
        model_name (str): LLM model identifier/name.
        settings (List[str]): Setting names to run.
        tasks (List[str]): Task names to run.
        alert_reps (List[str]): Alert representation options.
        kg_reps (List[str]): KG representation options.
        modality_to_exclude_desc (str): Description of modality exclusion.
        write_responses (bool): Whether to write results to file.
        results_dir (str): Directory where outputs should be written.
        select_samples_df (pd.DataFrame | None): Optional selective rerun filter.
        start_idx (int): Start index into `fault_gt`.
        stop_idx (int | None): Stop index (exclusive).
        context_length (int): Context length used when initializing models.
        reasoning_effort (str | None): OpenRouter reasoning effort for hybrid models.
        reasoning_max_tokens (int | None): Max reasoning tokens (overrides effort when set).

    Returns:
        None
    """
    results = {}
    entity_specifications, relationship_specifications = load_graph_entity_relationship_specifications(config["er_specifications_file"])
    
    for task in tasks:
        logging.info(f"----------------------------------------------------------------------------------------------------")
        logging.info(f"Running {task} ...")
        
        # Load task-specific parameters/info
        structured_output_type = config[task]['structured_output']
        structured_output_class = globals().get(structured_output_type)
        prompts_dir = config[task]["prompts_dir"]
        prompt_templates = load_prompt_templates(prompts_dir, None) # Load task-specific prompt templates for ALL settings
        
        for kg_representation in kg_reps:
            for alert_representation in alert_reps:
                for setting_name in settings:
                    setting_class_name = config[setting_name]['class_name']
                    setting_class = globals().get(setting_class_name)
                    
                    logging.info(f"--------------------------------------------------------------------------")
                    logging.info(f"Running {setting_name} with class {setting_class_name}...")
                    logging.info(f"Parameters: alert_rep = {alert_representation}, kg_rep = {kg_representation}, modalities = {modality_to_exclude_desc}")

                    setting_prompt_templates = {**prompt_templates,
                                                "rca": prompt_templates["rca"].get(setting_name, {})}

                    # Route to appropriate model loader based on model name
                    if '/' in model_name:  # OpenRouter models use provider/model format
                        model = load_model_openrouter(
                            model_name,
                            max_new_tokens=config[setting_name]['max_new_tokens'],
                            num_ctx=context_length,
                            reasoning_effort=reasoning_effort,
                            reasoning_max_tokens=reasoning_max_tokens
                        )
                    else:
                        model = load_model_ollama(model_name, max_new_tokens=config[setting_name]['max_new_tokens'], num_ctx=context_length)
                    test_results_dir = os.path.join(results_dir, task, modality_to_exclude_desc, kg_representation, alert_representation)
                    
                    for kg_file in kg_files:
                        logging.info(f"Executing scenarios for KG file: {kg_file}")
                        try:
                            run_test(
                                config,
                                setting_name,
                                setting_class,
                                kg_file,
                                fault_data,
                                fault_gt,
                                structured_output_class,
                                alert_representation,
                                kg_representation,
                                modality_to_exclude_desc,
                                model,
                                setting_prompt_templates,
                                entity_specifications,
                                relationship_specifications,
                                write_responses,
                                test_results_dir,
                                select_samples_df,
                                start_idx,
                                stop_idx
                            )
                        except Exception as e:
                            logging.error(
                                f"Error during evaluation of {task}, {setting_name}, {kg_file} with model={model.name}, kg_rep={kg_representation}, alert_rep={alert_representation}",
                                exc_info=True
                            )            
                    logging.info("---")
                    logging.info(f"Completed setting: {setting_name}")

def main():
    """
    CLI entry point to run RCA inference.

    Parses arguments, configures logging, loads systems/KGs/fault telemetry,
    and dispatches `run_tests` with the appropriate model and settings.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="RCA")
    parser.add_argument('--config', type=str, default="code/config/rca-config.yaml", help="Path to configuration yaml.")
    parser.add_argument('--system-name', type=str, help="Name of the system to test with. Select from 'MicroSS' or 'Online-Boutique'.")
    parser.add_argument('--kg-file', type=str, help="Path to the knowledge graph JSON file.")
    parser.add_argument('--fault-data-dir', type=str, help="Path to directory with fault data.")

    parser.add_argument('--model', type=str, help="LLM to use for evaluation.")
    parser.add_argument('--settings', type=str, help="Settings to run. Select from 'model_only', 'model_react', 'plan_execute_react'.")
    parser.add_argument('--tasks', type=str, help="Tasks to run. Select from 'cause_to_effect' and 'effect_to_cause'")
    parser.add_argument('--not-write-responses', action='store_false', help="Flag to not write responses to file. If excluded, responses will be written.")
    parser.add_argument('--results-dir', type=str, help="Directory to wite responses to file.")
    parser.add_argument('--log-file', type=str, help="Path to the execution log file")
    parser.add_argument('--start-index', type=int, help="First fault index to execute" )
    parser.add_argument('--stop-index', type=int, help="Stop fault index. This index does NOT get executed.")
    parser.add_argument('--context-length', type=int, help="The context length of the LLM model.")
    parser.add_argument('--alert-rep', type=str, help="How the alerts should be represented. Select from 'by-time' or 'by-component'")
    parser.add_argument('--kg-rep', type=str, help="How the KG should be represented. Select from 'list' or 'json'")
    parser.add_argument('--exclude-modalities', type=str, help="The alert modalities to exclude. Select from: 'logs', 'metrics', 'traces'")
    parser.add_argument('--rerun-samples-csv', type=str, help="Path to CSV listing samples to run (selective rerun). When omitted, all samples are executed.")
    parser.add_argument('--reasoning-effort', type=str, choices=['low', 'medium', 'high'], help="Reasoning effort level for OpenRouter hybrid models (e.g., Claude Opus 4.5). Options: low, medium, high.")
    parser.add_argument('--reasoning-max-tokens', type=int, help="Max tokens for reasoning/thinking on OpenRouter hybrid models. Overrides --reasoning if set.")
    args = parser.parse_args()
    start_time = time.time()
    
    logger = setup_logger(args.log_file)
    logger.info("Starting RCA analysis...")
    
    try:
        config = load_config(args.config)

        select_samples_df = None
        if args.rerun_samples_csv:
            print("Loading selective rerun samples CSV...")
            select_samples_df = pd.read_csv(args.rerun_samples_csv)
            print(f"Loaded {len(select_samples_df)} selective samples.")

        write_responses = args.not_write_responses
        start_idx = args.start_index if args.start_index is not None else config['start_index']
        stop_idx = args.stop_index or config['stop_index']
        modalities_to_exclude = args.exclude_modalities.split(" ") if args.exclude_modalities else [ None ]
        
        systems = args.system_name.split(" ") if args.system_name else config["systems"]
        for system in systems:
            kg_files = [args.kg_file] if args.kg_file else retrieve_files_from_directory(config["kg_file_dir"].format(system=system))
            for modality_to_exclude in modalities_to_exclude:
                modality_exclusion_desc = "all-modalities" if not modality_to_exclude else f"{modality_to_exclude}-excluded"
                
                # Load fault telemetry dataset for the system
                fault_data_dir = args.fault_data_dir if args.fault_data_dir else config['fault_data_dir'].format(system=system)
                fault_data = load_fault_data(
                    os.path.join(fault_data_dir, 'log/logs.json'),
                    os.path.join(fault_data_dir, 'trace/traces.json'),
                    os.path.join(fault_data_dir, 'metric/metrics.json'),
                    modality_to_exclude
                )
                fault_data = process_fault_data(fault_data, config[system]['alert_timezone'])
                fault_gt = pd.read_csv(
                    os.path.join(fault_data_dir, 'label.csv'),
                    header=0
                )
                
                # Other execution parameters
                models = [args.model] if args.model else config["models"]
                context_length = args.context_length or config["context_length"]
                settings = args.settings.split(" ") if args.settings else config['settings_to_run']
                tasks = args.tasks.split(" ") if args.tasks else config['tasks_to_run']
                alert_reps = args.alert_rep.split(" ") if args.alert_rep else config['alert_reps_to_run']
                kg_reps = args.kg_rep.split(" ") if args.kg_rep else config['kg_reps_to_run']
                config[LABEL_CURRENT_SYSTEM] = system
                
                for model_name in models:
                    results_directory = args.results_dir if args.results_dir else config["results_directory"].format(system=system, model=model_name)
                    # model = load_model_ollama(model_name) # load later with setting-specific parameters
                    run_tests(
                        config,
                        kg_files,
                        fault_data,
                        fault_gt,
                        model_name,
                        settings,
                        tasks,
                        alert_reps,
                        kg_reps,
                        modality_exclusion_desc,
                        write_responses,
                        results_directory,
                        select_samples_df,
                        start_idx,
                        stop_idx,
                        context_length,
                        args.reasoning_effort,
                        args.reasoning_max_tokens
                    )
    except Exception as e:
        logging.error(f"ERROR: Script crashed or exited early!", exc_info=True)
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        logging.info("--------- RCA COMPLETE ------------")
        logging.info(f"End time: {end_time}. Total time taken: {minutes} minutes {seconds} seconds")

if __name__ == "__main__":
    main()