"""
Trace Alert Extraction with Sliding Windows and Isolation Forest

Purpose:
- Window trace spans per flow (source, destination, operation) and compute features
- Detect anomalies using trained Isolation Forest detectors per feature
- Emit alert tuples for downstream RCA processing

Notes:
- Windows use TRACE_WINDOW_SIZE by default; stride controls overlap
- Features per window: mean duration, count of 500 responses, count of 400 responses
"""
import numpy as np
import pandas as pd

TRACE_WINDOW_SIZE = 30 * 1000 # 30 seconds in milliseconds

def slide_window(df, win_size=TRACE_WINDOW_SIZE, stride=None):
    """
    Slide a time window over trace data and extract per-window features.

    Args:
        df (pd.DataFrame): Columns start_time, end_time, status_code.
        win_size (int): Window size in milliseconds.
        stride (int | None): Step size in milliseconds. If None, equals win_size (no overlap).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (start_times, mean_durations, 500_counts, 400_counts).
    """
    if stride is None:
        stride = win_size

    df = df.copy()
    df['duration'] = df['end_time'] - df['start_time']
    t, time_max = df['start_time'].min(), df['start_time'].max()

    start_times, mean_durations, err_500_counts, err_400_counts = [], [], [], []

    while t < time_max:
        df_window = df[(df['start_time'] >= t) & (df['start_time'] < t + win_size)]
        if not df_window.empty:
            start_times.append(t)
            mean_durations.append(df_window['duration'].mean())
            err_500_counts.append((df_window['status_code'] == 500).sum())
            err_400_counts.append((df_window['status_code'] == 400).sum())
        t += stride

    return (
        np.array(start_times),
        np.array(mean_durations),
        np.array(err_500_counts),
        np.array(err_400_counts),
    )

def extract_trace_alerts(
    trace_df: pd.DataFrame,
    trace_detector: dict,
    win_size=TRACE_WINDOW_SIZE,
    stride=(TRACE_WINDOW_SIZE // 2) # Integer division - ensure true multiple.
) -> list[list]:
    """
    Extract alerts using trained Isolation Forest detectors per trace flow.

    For each (source, destination, operation) flow, apply sliding windows and
    compute features: mean duration, count of 500s, count of 400s. Each feature
    has its own IF detector; the first detected anomaly yields one alert per
    detector type for the flow. Duplicate alerts per (src, dst, op, type)
    are suppressed.

    Args:
        trace_df (pd.DataFrame): Columns: start_time, end_time, status_code, operation, parent_name, service_name, and timestamp.
        trace_detector (dict[str, dict]): Mapping from "<source>-<destination>-<operation>" to detectors with keys dur_detector, 500_detector, 400_detector.
        win_size (int): Window size in milliseconds.
        stride (int): Step size in milliseconds (default half window).

    Returns:
        list[list]: Alerts as [timestamp, source, destination, operation, anomaly_type] sorted by timestamp.
    """
    if trace_df.empty:
        return []
    
    trace_df = trace_df.sort_values('timestamp')
    trace_df['operation'] = trace_df['operation'].str.split('?').str[0]
    grouped = trace_df.groupby(['parent_name', 'service_name', 'operation'])
    
    alerts = []
    seen_alert_keys = set() # (src, dst, op, typ) 

    for (source, destination, operation), call_df in grouped:
        flow_key = f"{source}-{destination}-{operation}"
        if flow_key not in trace_detector:
            continue
        
        start_times, durations, err_500, err_400 = slide_window(call_df, win_size, stride)
        
        if len(start_times) == 0:
            continue

        detectors = trace_detector[flow_key]
        for detector_type, alert_type, values in [('dur', 'PD', durations), 
                                                  ('500', '500', err_500), 
                                                  ('400', '400', err_400)]:
            timestamp = iforest(detectors[f'{detector_type}_detector'], values, start_times)
            if timestamp is None:
                continue
            
            key = (source, destination, operation, detector_type)
            if (key in seen_alert_keys):
                continue
            seen_alert_keys.add(key)
            
            alerts.append([str(timestamp), source, destination, operation, alert_type])

    return sorted(alerts, key=lambda x: int(x[0]))


def iforest(detector, values, timestamps) -> int | None:
    """
    Run an Isolation Forest and return the timestamp of the first anomaly.

    Args:
        detector: Trained IsolationForest instance supporting .predict().
        values (np.ndarray): Feature values per window.
        timestamps (np.ndarray): Corresponding window start times.

    Returns:
        int | None: Timestamp of the first detected anomaly, or None if no anomalies.
    """
    if len(values) == 0:
        return None

    labels = detector.predict(values.reshape(-1, 1))
    anomaly_indices = np.where(np.array(labels) == -1)[0]
    if len(anomaly_indices) == 0:
        return None
    
    return timestamps[anomaly_indices[0]]