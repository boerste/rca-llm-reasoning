from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import pytz

@dataclass
class LogAlert:
    """
    Represents a log alert.

    Attributes:
        component (str): The component that generated the log.
        id (str): Unique identifier for the log pattern/template.
        messages (list[str]): List of raw log messages associated with this alert.
        count (int): Number of times this log pattern occurred.
        repr_message (Optional[str]): A representative message for the log pattern.
        timestamp (Optional[datetime]): Timestamp of the first occurrence.
        all_timestamps (Optional[list[datetime]]): Timestamps of all occurrences.
        mean_freq (Optional[float]): Mean frequency of occurrence in seconds.
    """
    component: str
    id: str
    messages: list[str]
    count: int
    repr_message: Optional[str] = None # A representative message if count is high
    
    # Derived fields
    timestamp: Optional[datetime] = field(init=False, default=None)
    all_timestamps: Optional[list[datetime]] = field(init=False, default=None)
    mean_freq: Optional[float] = field(init=False, default=None)
    
    def __post_init__(self):
        # Extract timestamps from messages
        timestamps = sorted([
            datetime.strptime(msg.split(" |")[0].replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")
            for msg in self.messages
        ])
        self.timestamp = timestamps[0]
        self.all_timestamps = timestamps
        
        if self.repr_message:
            # Compute time deltas
            deltas = [
                (t2 - t1).total_seconds()
                for t1, t2 in zip(timestamps[:-1], timestamps[1:])
            ]
            self.mean_freq = sum(deltas) / len(deltas)
                
    def format_representative_message(self):
        """Formats the representative message with occurrence statistics."""
        return f"`{self.repr_message}` (occurred {self.count} times from {self.timestamp.strftime("%H:%M:%S.%f")[:-3]} to {self.all_timestamps[-1].strftime("%H:%M:%S.%f")[:-3]} approx every {self.mean_freq:.3f}s, representative shown)"
    
    def format_for_kg(self):
        """Formats the log alert for the knowledge graph."""
        if self.repr_message is not None:
            str_repr = self.format_representative_message()
        else:
            str_repr = "\n".join(self.messages)
        return str_repr
    
    def format_all(self):
        """Formats the log alert for textual display with all attributes."""
        if self.repr_message is not None:
            # Format representative message and exclude timestamp
            message_repr = f"`{self.format_representative_message().split("| ", 1)[1]}"
        else:
            messages = []
            for msg, ts in zip(self.messages, self.all_timestamps):
                messages.append(f"{ts.strftime("%H:%M:%S.%f")[:-3]}: `{msg.split("| ", 1)[1]}`")
            message_repr = " >>> ".join(messages)
            
        return f"{self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]} | LOG | {self.component} | {message_repr}"
        
@dataclass
class MetricAlert:
    """
    Represents a metric alert.

    Attributes:
        timestamp (str): Timestamp of the alert.
        component (str): The component (e.g., service, host) reporting the metric.
        host (str): The host where the component is running.
        metric (str): The name of the metric.
        direction (str): The direction of the anomaly (e.g., up, down).
    """
    timestamp: str
    component: str
    host: str
    metric: str
    direction: str
    
    def format_for_kg(self):
        """Formats the metric alert for the knowledge graph."""
        return f"{self.timestamp} | {self.metric}: {self.direction}"
    
    def format_all(self):
        """Formats the metric alert for textual display with all attributes."""
        return f"{self.timestamp} | METRIC | {self.component} | {self.metric} | {self.direction}"
        
@dataclass
class TraceAlert:
    """
    Represents a distributed trace alert.

    Attributes:
        timestamp (str): Timestamp of the alert.
        source (str): The source component of the trace alert.
        target (str): The target component of the trace alert.
        api (str): The API or operation name.
        type (str): The type of trace anomaly (e.g., latency, error).
    """
    timestamp: str
    source: str
    target: str
    api: str
    type: str
    
    def format_for_kg(self):
        """Formats the trace alert for the knowledge graph."""
        return f"{self.timestamp} | {self.source} --> {self.target} | {self.api} | {self.type} "
    
    def format_all(self):
        """Formats the trace alert for textual display with all attributes."""
        return f"{self.timestamp} | TRACE | {self.source} --> {self.target} | {self.api} | {self.type}"

@dataclass
class FaultScenario:
    """
    Represents a complete scenario including all alerts associated with a single fault.

    Attributes:
        id (int): Unique identifier of the fault.
        log_alerts (List[LogAlert]): List of log alerts.
        metric_alerts (List[MetricAlert]): List of metric alerts.
        trace_alerts (List[TraceAlert]): List of trace alerts.
    """
    id: int
    log_alerts: List[LogAlert]
    metric_alerts: List[MetricAlert]
    trace_alerts: List[TraceAlert]
    
    @staticmethod
    def from_dict(dict):
        """Create a scenario instance from a dictionary."""
        return FaultScenario(id = dict['id'],
                             log_alerts = dict['log_alerts'],
                             trace_alerts = dict['trace_alerts'],
                             metric_alerts = dict['metric_alerts'])

def get_possible_fault_types(system: str, as_dict = False, as_categorical = False, version = None):
    """
    Returns possible fault types for a given system.

    Args:
        system (str): The name of the system (e.g., 'MicroSS', 'Online-Boutique-cloudbed').
        as_dict (bool): If True, returns a dictionary mapping fault names to their normalized forms.
        as_categorical (bool): If True, returns a dictionary mapping entity types to lists of faults.
        version (Optional[str]): Version of the system (unused).

    Returns:
        Union[dict, str]: A dictionary or string representation of possible fault types.
    """
    if system == "MicroSS":
        if as_dict:
            return {
                'high memory usage': '[memory_anomalies]',
                'unexpected process termination': '[normal memory freed label]',
                'session timeout': '[login failure]',
                'file missing': '[file moving program]',
                'internal permission misconfiguration': '[access permission denied exception]',
                # other acceptable synonyms
            }
        elif as_categorical:
            return {
                "service_instance": [
                    '[memory_anomalies]',
                    '[normal memory freed label]',
                    '[login failure]',
                    '[file moving program]',
                    '[access permission denied exception]',
                ]
            }
        else:
            return (
                "  1. high memory usage\n"
                "  2. unexpected process termination\n"
                "  3. session timeout\n"
                "  4. file missing\n"
                "  5. internal permission misconfiguration"
            )
    
    elif "Online-Boutique-cloudbed" in system:
        if as_dict:
            return {
                'container cpu load': 'container CPU load',
                'container memory load': 'container memory load',
                'container network packet retransmission': 'container network packet retransmission',
                'container network packet corruption': 'container network packet corruption',
                'container network latency': 'container network latency',
                'container packet loss': 'container packet loss',
                'container process termination': 'container process termination',
                'container read i/o load': 'container read I/O load',
                'container write i/o load': 'container write I/O load',
                'node cpu load': 'node CPU load',
                'node cpu spike': 'node CPU spike',
                'node memory consumption': 'node memory consumption',
                'node disk read i/o consumption': 'node disk read I/O consumption',
                'node disk write i/o consumption': 'node disk write I/O consumption',
                'node disk space consumption': 'node disk space consumption',
                # other acceptable synonyms
                'container network packet loss': 'container packet loss',
                'container network packet latency': 'container network latency',
                'container read i o load': 'container read I/O load',
                'container read io load': 'container read I/O load',
                'container write i o load': 'container write I/O load',
                'container write io load': 'container write I/O load',
                'node disk read i o consumption': 'node disk read I/O consumption',
                'node disk read io consumption': 'node disk read I/O consumption',
                'node disk write i o consumption': 'node disk write I/O consumption',
                'node disk write io consumption': 'node disk write I/O consumption'

            }
        elif as_categorical:
            return {
                "pod": [
                    'container CPU load',
                    'container memory load',
                    'container network packet retransmission',
                    'container network packet corruption',
                    'container network latency',
                    'container packet loss',
                    'container process termination',
                    'container read I/O load',
                    'container write I/O load',
                ],
                "service": [
                    'container CPU load',
                    'container memory load',
                    'container network packet retransmission',
                    'container network packet corruption',
                    'container network latency',
                    'container packet loss',
                    'container process termination',
                    'container read I/O load',
                    'container write I/O load',
                ],
                "node": [
                    'node CPU load',
                    'node CPU spike',
                    'node memory consumption',
                    'node disk read I/O consumption',
                    'node disk write I/O consumption',
                    'node disk space consumption',
                ]
            }
        else:
            return (
                "  1. container CPU load\n"
                "  2. container memory load\n"
                "  3. container network packet retransmission\n"
                "  4. container network packet corruption\n"
                "  5. container network latency\n"
                "  6. container packet loss\n"
                "  7. container process termination\n"
                "  8. container read I/O load\n"
                "  9. container write I/O load\n"
                "  10. node CPU load\n"
                "  11. node CPU spike\n"
                "  12. node memory consumption\n"
                "  13. node disk read I/O consumption\n"
                "  14. node disk write I/O consumption\n"
                "  15. node disk space consumption\n"
                " Container-level faults (1-9) can occur at the `Service_Instance` or `Service` level."
                " A fault at the `Service` level typically indicates that multiple, if not all, instances of that service are affected (e.g., due to shared configurations, dependencies, or systemic resource constraints)."
                " Node-level faults (10-15) occur only on `Host` entities."
            )
    else:
        return {} if as_dict else ""

def get_possible_entity_types_for_root_cause_location(system: str, as_list = False):
    """
    Returns possible entity types where a root cause can occur for a given system.

    Args:
        system (str): The name of the system.
        as_list (bool): If True, returns a list of entity types.

    Returns:
        Union[list, str]: A list or string representation of entity types.
    """
    if system == "MicroSS":
        if as_list:
            return ['Service_Instance']
        else:
            return "`Service_Instance`"
    elif "Online-Boutique-cloudbed" in system:
        if as_list:
            return ['Service_Instance', 'Service', 'Host']
        else:
            return "`Service_Instance`, `Service`, or `Host`"
    else:
        return []
    
def is_alerts_non_empty_for_fault(alerts: dict) -> bool:
    """Check whether alerts dict contains any alerts."""

    if 'by_component' in alerts.keys() and 'by_edge' in alerts.keys():
        if any(alerts['by_component'].values()) or any(alerts['by_edge'].values()):
            return True
    else:
        if any(alerts.values()):
            return True
    return False

# --------------- Alert pre-processing ---------------

def convert_epoch_to_datetime_str(ts: str, timezone = 'UTC') -> Optional[str]:
    """
    Converts an epoch timestamp string (in seconds or milliseconds) to a datetime 
    string in the given timezone.

    Args:
        ts (str): Epoch time as a string (seconds or milliseconds).
        timezone (str): Timezone name (e.g., 'UTC', 'Asia/Brunei').

    Returns:
        str: Datetime string formatted as 'YYYY-MM-DD HH:MM:SS.sss', or None if invalid.
    """
    try:
        tz = pytz.timezone(timezone)
        ts_float = float(ts)
        if ts_float > 1e12:  # milliseconds
            dt = datetime.fromtimestamp(ts_float / 1000.0, tz=tz)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # trim to ms
        elif ts_float > 1e9:  # seconds
            dt = datetime.fromtimestamp(ts_float, tz=tz)
            return dt.strftime('%Y-%m-%d %H:%M:%S')  
    except (ValueError, TypeError):
        return None

def convert_timestamps_for_alert(alerts, alert_timezone):
    """
    Converts alert timestamps from epoch to datetime string in given timezone.
    
    Args:
        alerts (dict): The dictionary of alerts.
        alert_timezone (str): The timezone for conversion.
        
    Returns:
        dict: The updated alerts dictionary with converted timestamps.
    """
    for alert_list_key in ["metric_alerts", "trace_alerts", "log_alerts"]:
        alert_list = alerts.get(alert_list_key, [])

        # Normalize timestamps -- keep existing format if not epoch
        for alert in alert_list:
            converted = convert_epoch_to_datetime_str(alert.timestamp, alert_timezone)
            if converted:
                alert.timestamp = converted
    
    return alerts

def remove_alert_noise(alerts):
    """
    Filters out irrelevant trace alerts and normalizes component names in metric alerts.

    Args:
        alerts (dict): The dictionary of alerts to process.

    Returns:
        dict: The filtered and normalized alerts.
    """

    # Filter trace alerts that have the same source & target and belong to 'telemetry' signals
    filtered_trace_alerts = []
    trace_alert: TraceAlert
    for trace_alert in alerts['trace_alerts']:
        if trace_alert.source == trace_alert.target:
            continue
        if '(telemetry)' in trace_alert.api:
            continue
        filtered_trace_alerts.append(trace_alert)
    alerts['trace_alerts'] = filtered_trace_alerts

    # Normalize metric component names
    rename_map = {
        "0.0.0.1": "host1",
        "0.0.0.2": "host2",
        "0.0.0.3": "host3",
        "0.0.0.4": "host4"
    }
    metric_alert: MetricAlert
    for metric_alert in alerts['metric_alerts']:
        if metric_alert.component in rename_map.keys() and metric_alert.host == "system":
            metric_alert.component = rename_map[metric_alert.component]

    return alerts
    
def sort_alerts_for_fault(alerts, keys=[]):
    """
    Sorts metric, trace, and log alerts separately by timestamp and
    sorts 'all' alerts by timestamp.
    
    Args:
        alerts (dict): The dictionary of alerts.
        keys (list): List of keys to sort (e.g., ["metric_alerts", "trace_alerts", "log_alerts", "all"]).
        
    Returns:
        dict: The sorted alerts dictionary.
    """
    if not keys:
        keys = ["metric_alerts", "trace_alerts", "log_alerts", "all"]
    for alert_list_key in keys:
        # Alerts can be nested, e.g., by component
        alert_list = alerts.get(alert_list_key, [])
        
        # Sort by timestamp
        try:
            alert_list.sort(
                key=lambda a: a.timestamp if isinstance(a.timestamp, datetime) else datetime.strptime(a.timestamp, '%Y-%m-%d %H:%M:%S.%f')
            )
        except ValueError:
            # If any timestamp is not in the expected format, skip sorting
            print(f"Timestamp not in expected format: {alert_list}")
            pass

    return alerts

def organize_alerts_for_fault_by_component(alerts):
    """
    Organize fault alerts by component (metrics and logs) and edge (traces).
    
    Args:
        alerts (dict): The dictionary of alerts to organize.

    Returns:
        dict: The alerts dictionary augmented with 'by_component' and 'by_edge' keys.
    """
    by_component = {}
    by_edge = {}

    # Organize logs by component
    log_alert: LogAlert
    for log_alert in alerts["log_alerts"]:
        comp = log_alert.component
        if comp not in by_component:
            by_component[comp] = {"log_alerts": [], "metric_alerts": [], "all": []}
        by_component[comp]["log_alerts"].append(log_alert)
        by_component[comp]["all"].append(log_alert)

    # Organize metrics by component
    metric_alert: MetricAlert
    for metric_alert in alerts["metric_alerts"]:
        comp = metric_alert.component
        if comp not in by_component:
            by_component[comp] = {"log_alerts": [], "metric_alerts": [], "all": []}
        by_component[comp]["metric_alerts"].append(metric_alert)
        by_component[comp]["all"].append(metric_alert)

    # Organize traces by edge (source, target)
    trace_alert: TraceAlert
    for trace_alert in alerts["trace_alerts"]:
        edge = (trace_alert.source, trace_alert.target)
        if (edge) not in by_edge:
            by_edge[edge] = []
        by_edge[edge].append(trace_alert)

    # Sort alerts per component
    for comp in by_component.keys():
        by_component[comp] = sort_alerts_for_fault(by_component[comp])

    alerts['by_component'] = by_component
    alerts['by_edge'] = by_edge
    return alerts

def process_alerts_for_fault(alerts, alert_timezone):
    """
    Processes alerts for a SINGLE fault.
    Performs timestamp normalization, noise removal, sorting, and organization.

    Args:
        alerts (dict): Dictionary of raw alerts for a single fault.
        alert_timezone (str): Timezone for timestamp conversion.

    Returns:
        dict: Dictionary mapping fault IDs to processed alerts.
    """
    # Normalize alert timestamps
    alerts = convert_timestamps_for_alert(alerts, alert_timezone)

    # Remove alert noise
    alerts = remove_alert_noise(alerts)

    alerts['all'] = alerts['log_alerts'] + alerts['metric_alerts'] + alerts['trace_alerts']
    
    # Sort alerts (modality-wise and all)
    alerts = sort_alerts_for_fault(alerts)
    
    # Organize alerts by component
    alerts = organize_alerts_for_fault_by_component(alerts)
    
    return alerts

def process_fault_data(fault_data: dict, alert_timezone):
    """
    Processes alerts for MULTIPLE faults.

    Args:
        fault_data (dict): Dictionary mapping fault IDs to raw alerts.
        alert_timezone (str): Timezone for timestamp conversion.

    Returns:
        dict: Dictionary mapping fault IDs to processed alerts.
    """
    processed_fault_data = {}
    for fault_id, alerts in fault_data.items():
        alerts = process_alerts_for_fault(alerts, alert_timezone)
        processed_fault_data[fault_id] = alerts
    return processed_fault_data

# --------------- Alert formatting ---------------

def format_alerts_for_fault_by_time(alerts_for_fault: dict):
    """
    Format all alerts for a SINGLE fault as a string. 
    Alerts are presented temporally without additional grouping.
    
    Args:
        alerts_for_fault (dict): The processed alerts for a fault.
        
    Returns:
        str: The formatted string representation of the alerts.
    """
    
    if not 'all' in alerts_for_fault.keys():
        alerts_for_fault = process_alerts_for_fault(alerts_for_fault)
        
    by_time = "\n".join([f"- {alert.format_all()}" for alert in alerts_for_fault['all']])
    return by_time

def format_alerts_for_fault_by_component(alerts_for_fault: dict):
    """
    Format all alerts for a SINGLE fault as a string. 
    Group by component/edge.
    
    Args:
        alerts_for_fault (dict): The processed alerts for a fault.
        
    Returns:
        str: The formatted string representation of the alerts.
    """
    
    # If not yet organized by component and edge
    if not all(grouping in alerts_for_fault for grouping in ('by_component', 'by_edge')):
        alerts_for_fault = process_alerts_for_fault(alerts_for_fault)
    
    by_component = ""
    for comp in alerts_for_fault['by_component'].keys():
        by_component += f"- {comp}:\n"
        alerts = alerts_for_fault['by_component'][comp]['all']
        if len(alerts) == 0: continue
        alerts_str = "\n".join([f"  - {alert.format_all()}" for alert in alerts])
        by_component += f"{alerts_str} \n\n"
    if by_component == "":
        by_component = "No metric or log alerts were detected."
        
    by_edge = ""
    for edge, alerts in alerts_for_fault['by_edge'].items():
        source, target = edge
        alerts_str = "\n".join([f"  - {alert.format_all()}" for alert in alerts])
        by_edge += f"- {source} --> {target}:\n{alerts_str} \n\n"
    if by_edge == "":
        by_edge = "No trace alerts were detected."
            
    return f"{by_component}\n\n{by_edge}"

def format_alerts_for_fault(alerts_for_fault: dict, strategy: str):
    """
    Format all alerts for fault as a string based on the provided strategy.
    
    Args:
        alerts_for_fault (dict): The processed alerts for a fault.
        strategy (str): The formatting strategy ('by-time' or 'by-component').
        
    Returns:
        str: The formatted string representation of the alerts.
    """
    if strategy == 'by-time':
        return format_alerts_for_fault_by_time(alerts_for_fault)
    elif strategy == 'by-component':
        return format_alerts_for_fault_by_component(alerts_for_fault)
    else:
        raise ValueError("Strategy must be one of 'by-time' or 'by-component'")