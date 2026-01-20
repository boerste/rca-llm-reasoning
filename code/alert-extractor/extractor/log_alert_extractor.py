"""
Log Alert Extraction via Drain Templates

Purpose:
- Match raw logs to Drain3 templates and compute template-level alerts
- Select alerts via low-frequency templates and error-keyword templates
- Group alerts per (template_id, entity), with optional timestamp prefixing

Notes:
- Uses a pre-trained TemplateMiner from Drain3; see drain utilities in
    `alert-extractor/drain/drain_template_extractor.py` for miner training/persistence.
"""
import pandas as pd
from drain.drain_template_extractor import *
from drain3 import TemplateMiner
from utils.time_util import convert_epoch_to_datetime_str

MAX_INDIVIDAL_LOGS_PER_TEMPLATE_ID = 5
INDEX_FOR_REPR_MESSAGE = 0

def processing_feature(entity: str, log: str, timestamp, miner) -> dict:  
    """
    Match a log line to a Drain template and return normalized fields.

    Assigns id = cluster_id when matched; -1 when unmatched. Removes
    trailing newline from the message and sets count = 1 as a unit record.

    Args:
        entity (str): Source entity/component of the log.
        log (str): Raw log message.
        timestamp (int | float): Epoch timestamp for the log.
        miner (TemplateMiner): Pre-trained Drain3 template miner.

    Returns:
        dict: Feature dict with keys: timestamp, entity, id, count, message.
    """ 
    cluster = miner.match(log)
    template_id = -1 if cluster is None else cluster.cluster_id
    return {
        'timestamp': timestamp,
        'entity': entity,
        'id': template_id,
        'count': 1,
        'message': log.rstrip('\n') # Current log message contains newline at the end. Remove.
    }

def extract_log_alerts(log_df: pd.DataFrame, miner: TemplateMiner, low_freq_p: float, max_logs_per_template: int = 5, add_time_to_log_message: bool = False, timezone: str = 'UTC') -> list[list]:
    """
    Extract log-based alerts by matching logs to Drain templates and filtering.

    Selection criteria:
    - Low-frequency templates (bottom low_freq_p quantile)
    - Templates whose template text contains error keywords ('error', 'errno', 'fail', 'exception')
    - Always include unmatched logs (id = -1)

    Groups alerts by (id, entity), aggregates messages/timestamps/count,
    and optionally adds a representative message when count exceeds a threshold.
    When add_time_to_log_message=True, prefixes ISO timestamps to messages
    using convert_epoch_to_datetime_str() with the given timezone.

    Args:
        log_df (pd.DataFrame): DataFrame with message, timestamp, and entity columns.
        miner (TemplateMiner): Pre-trained Drain3 TemplateMiner instance.
        low_freq_p (float): Fraction of least frequent templates to treat as alerts (0–1).
        max_logs_per_template (int): Threshold for including a representative message.
        add_time_to_log_message (bool): If True, prefix formatted time to messages.
        timezone (str): Timezone used for timestamp formatting; default 'UTC'.

    Returns:
        list[list]: Alert entries of shape [entity, template_id, messages, count, representative_message].
    """
    # Early exit if no logs to process
    if log_df.empty:
        return []
    
    all_clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=False)
    error_keywords = ['error', 'errno', 'fail', 'exception']
    
    selected_template_ids = set()
    for idx, cluster in enumerate(all_clusters):
        if idx < int(low_freq_p * len(all_clusters)):
            # Low-frequency templates
            selected_template_ids.add(cluster.cluster_id)
        elif any(keyword in cluster.get_template().lower() for keyword in error_keywords):
            # Error log keywords
            selected_template_ids.add(cluster.cluster_id)
    selected_template_ids.add(-1) # Include unmatched logs

    # Match each log to its template
    log_df = log_df.sort_values(by='timestamp', ascending=True)
    log_features = [processing_feature(entity, message, timestamp, miner) 
                    for entity, message, timestamp in zip(log_df['entity'], log_df['message'], log_df['timestamp'])]
    feature_df = pd.DataFrame(log_features)
    
    # Filter by selected templates
    alert_df = feature_df[feature_df['id'].isin(selected_template_ids)]
    
    # Group alerts based by id and entity
    alert_groups = (
        alert_df
        .groupby(['id', 'entity'])
        .agg({
            'message': list,
            'timestamp': list,
            'count': 'sum'
        })
        .reset_index()
    )
    
    # Add representative message (i.e., first in list of messages) if count is greater than max allowed or all messages are the same
    def get_representative_message(row):
        messages = row['message']
        # Optionally: or len(set(messages)) == 1
        return messages[INDEX_FOR_REPR_MESSAGE] if (row['count'] > max_logs_per_template) else None
    
    alert_groups['representative_message'] = alert_groups.apply(get_representative_message, axis=1)
    
    if add_time_to_log_message:
        # Prepend formatted timestamp to each log message
        def prepend_time(row):
            ts_list = row['timestamp']
            msg_list = row['message']
            if row['count'] > 0:
                return [f"{convert_epoch_to_datetime_str(ts, timezone)} | {msg}" for ts, msg in zip(ts_list, msg_list)]
            else:
                return None
        
        def prepend_time_repr_message(row):
            if pd.notna(row['representative_message']):
                msg = row['representative_message']
                ts = row['timestamp'][INDEX_FOR_REPR_MESSAGE]
                return f"{convert_epoch_to_datetime_str(ts, timezone)} | {msg}"
            else:
                return None
        
        alert_groups['message'] = alert_groups.apply(prepend_time, axis=1)
        alert_groups['representative_message'] = alert_groups.apply(prepend_time_repr_message, axis=1)
    
    alerts=[
        [row['entity'], str(row['id']), row['message'], row['count'], row['representative_message']]
        for _, row in alert_groups.iterrows()
    ]

    return alerts

