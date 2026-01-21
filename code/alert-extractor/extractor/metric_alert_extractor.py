"""
Metric Alert Extraction via k-sigma Detection

Purpose:
- Scan KPI time series per component and flag anomalies using the k-sigma rule
- Map anomalies into alert tuples compatible with downstream RCA processing

Notes:
- Expects per-component KPI frames with 'metric', 'timestamp', and 'value'
- Dataset-specific parsing extracts 'entity' and 'host' from the component id
"""
def extract_metric_alerts(component: str, kpi_df: dict, metric_detector: dict, dataset: str, sigma_k: int = 3) -> list[list]:
    """
    Extract metric alerts using a k-sigma (default 3-sigma) anomaly detector.

    For each KPI in the component's time series, flags the first point outside
    the interval [mu - k*sigma, mu + k*sigma] and emits an alert with
    direction. Component id is parsed into (entity, host) depending on dataset.

    Args:
        component (str): Component id (e.g., GAIA: webservice1_0.0.0.2; Market: node-6.adservice-2).
        kpi_df (pd.DataFrame): Per-component data with columns metric, timestamp, value.
        metric_detector (dict[str, tuple[float, float]]): KPI mean/std by name.
        dataset (str): Dataset name controlling (entity, host) parsing.
        sigma_k (int): Multiples of std for k-sigma threshold; defaults to 3.

    Returns:
        list[list]: Alerts as [timestamp, entity, host, kpi, direction].
    """
    alerts = []
    
    for kpi, df in kpi_df.groupby('metric'):
        if kpi not in metric_detector or df.empty:
            continue
        df = df.fillna(0)
        df = df.sort_values(by="timestamp", ascending=True)

        times = df['timestamp'].values
        values = df['value'].values
        
        ab_index, direction = k_sigma(metric_detector[kpi], values, k=sigma_k)
        
        if ab_index != -1:
            ab_time = times[ab_index]
            if dataset == "GAIA":
                splits = component.split('_')
                if "system" in splits:
                    # e.g. component = system_0.0.0.1
                    entity, host = splits[1], splits[0]
                else:
                    entity, host = splits[0], splits[1]
            elif "OpenRCA-Market" in dataset:
                if "." in component:
                    splits = component.split('.')
                    entity, host = splits[1], splits[0] 
                else:
                    entity, host = component, None
            else:
                entity, host = None, None
                
            alerts.append([ab_time, entity, host, kpi, direction])
               
    return alerts


def k_sigma(detector, test_arr, k=3) -> tuple[int, str | None]:
    """
    Apply the k-sigma rule to flag the first out-of-bound value.

    Uses thresholds upper = mean + k*std and lower = mean - k*std.

    Args:
        detector (tuple[float, float]): (mean, std) for the KPI.
        test_arr (Sequence[float]): KPI values to test in order.
        k (int): Sigma multiplier for thresholds; defaults to 3.

    Returns:
        tuple[int, str | None]: (index, direction) where index is the first
        anomaly index or -1 if none; direction is 'up', 'down', or None.
    """
    mean, std = detector
    upper = mean + k * std
    lower = mean - k * std

    for index, value in enumerate(test_arr):
        if value > upper:
            return index, 'up'
        elif value < lower:
            return index, 'down'

    return -1, None
