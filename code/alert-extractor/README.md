
# Alert Extraction 
 
Alert Extraction turns raw telemetry and injected-fault labels into compact metric, trace, and log alerts for downstream root-cause analysis. It unifies telemetry, learns a baseline of normal behavior, mines log templates, and detects anomalies per fault.

## Pipeline

1. **Dataset download**: download each respective dataset locally (linked below).
2. **Dataset prep**: normalize/standardize labels using the appropriate prep script
    - [prep_GAIA_dataset.py](prep_GAIA_dataset.py)
    - [prep_OpenRCA_Market_dataset.py](prep_OpenRCA_Market_dataset.py)
    - [prep_OpenRCA_Telecom_dataset.py](prep_OpenRCA_Telecom_dataset.py)
3. **Fault telemetry prep**: process telemetry and slice pre- and post-fault windows with [extract_pre_and_post_fault_telemetry.py](extract_pre_and_post_fault_telemetry.py).
4. **Train detectors**: 
    - Fit metric and trace detectors with [train_metric_trace_detectors.py](train_metric_trace_detectors.py).
    - Train Drain3 template miner for logs with [train_log_template_miner.py](train_log_template_miner.py).
5. **Alert generation**: produce final metric, trace, and log alerts with [alert_extractor.py](alert_extractor.py).

**Note: Steps 2–5 above are automated by [extract_alerts.sh](../../extract_alerts.sh).**

The [config.yaml](config.yaml) file contains the per-dataset config for alert extraction. The key parameters are:
- `telemetry-data`: path to the raw dataset
- `processed-data`: path to location where processed data can be saved
- `fault-alert-data`: path to location where fault alerts should be saved
- `gt-file`: path to the labelled ground-truth with info about injected faults


## GAIA
### Dataset
We use the `MicroSS` telemetry data found in [GAIA-DataSet](!https://github.com/CloudWise-OpenSource/GAIA-DataSet).

### Injected faults (2021/07)
We only consider faults injected in 2021/07 because the telemetry data from 2021/08 does not contain trace data. Addtionally, log data is missing from 2021-07-23 09:28:23.476 onwards in 2021/07, so we exclude this period as well. 

The procedure for selecting the injected faults considered for evaluation can be found in [`prep_GAIA_dataset_labels.py`](./prep_GAIA_dataset_labels.py). It consists of two high-level steps:
1. For each selected fault, we ensure that no telemetry outages exist for the core service instances. 
2. Due to the high imbalance of injected faults for `[file moving program]`, `[normal memory freed label]`, and `[access permission denied exception]`, we downsample for `[login failure]` and `[memory_anomalies]` in an effort to better balance the inject fault type distribution. The final distribution is as follows:

| Injected Failure                     | Count | Avg Duration (s) | Fault Type
|--------------------------------------|-------|------------------|------------
| [login failure]                      | 40    | 11               | session timeout
| [memory_anomalies]                   | 40    | 600              | high memory usage
| [file moving program]                | 36    | 600              | file missing
| [normal memory freed label]          | 20    | 600              | unexpected process termination
| [access permission denied exception] | 16    | 3600             | internal permission misconfiguration

### Extraction details
The ground-truth labelled fault-injection file is [label.csv](../../data/fault-alerts/MicroSS/label.csv).
The extracted alerts can be found in [metrics.json](../../data/fault-alerts/MicroSS/metric/metrics.json), [traces.json](../../data/fault-alerts/MicroSS/trace/traces.json), and [logs.json](../../data/fault-alerts/MicroSS/log/logs.json).

Each alert is a tuple consisting of:
- Metrics: _timestamp_, _entity_, _host_, _metric name_, _anomaly direction_
    - The _timestamp_ corresponds to the occurrence of the **first** detected anomaly value of the metric in the fault window.
- Traces: _timestamp_, _callee_, _caller_, _operation name_, _abnormality type_ (i.e., PD, 400, 500)
    - The _timestamp_ corresponds to the occurrence of the **first** detected anomaly value in the fault window.
- Logs: _entity_, _log id_, _messages_, _count_, _representative message_
    - The _representative message_ field is populated as the first message in the list of messages if the _count_ is greater than `MAX_INDIVIDAL_LOGS_PER_TEMPLATE_ID = 5`.

#### Note on timestamps
The telemetry data does not contain uniform time representations: the metrics are provided using Unix epoch time, whereas the logs and traces are presented as date-time strings. Though not mentioned in the documentation, we deduce that the trace and log date-time strings are GMT+08 based on correlations between the metric data values and known injected faults.

#### Note on `loginservice`
The MicroSS documentation states that there are two instances of a `loginservice`. In the telemetry data, there is only reference to a `logservice`. Based on the log messages and trace operations (`db_login_methods`, `login_model_implement`, `login_query_redis_info`), it leads us to believe these in fact mean the `loginservice`.

Therefore, post fault extraction, we change all instances of `logservice` to `loginservice` in `metrics.json`, `traces.json`, `logs.json`, and `label.csv`. 
Additionally, the host information of `loginservice1` and `loginservice2` from the documentation is swapped when looking closely at the log and trace information (i.e., `loginservice1` is hosted on `host3`, and `loginservice2` is hosted on `host 2`, instead of vice versa). The [MicroSS KG](../../data/system-knowledge-graphs/original/MicroSS/MicroSS-KG.json) reflects this.

## OpenRCA-Market
### Dataset
We use the `cloudbed-1` and `cloudbed-2` datasets from [OpenRCA Market](!https://github.com/microsoft/OpenRCA). This dataset is a cleaned-up version of the AIOps 2022 Challenge Dataset ([dataset - link may not work](!https://competition.aiops-challenge.com/home/competition/1496398526429724760), [instructions](!https://www.bizseer.com/index.php?m=content&c=index&a=show&catid=25&id=83)).

### Injected faults
| Fault Type                                 | Count |
|--------------------------------------------|--------|
| container read I/O load                    | 17     |
| container CPU load                         | 13     |
| container memory load                      | 13     |
| container network packet corruption        | 13     |
| container network packet retransmission    | 13     |
| node disk space consumption                | 10     |
| node disk write I/O consumption            | 10     |
| node CPU load                              | 9      |
| node disk read I/O consumption             | 9      |
| node memory consumption                    | 9      |
| container network latency                  | 8      |
| container packet loss                      | 8      |
| container process termination              | 7      |
| container write I/O load                   | 5      |
| node CPU spike                             | 4      |

From the challenge instructions, we surmise that each fault is detectable in the telemetry data within 10 minutes from the injection time. We therefore use a 10 minutes window as the anomalous window for each injected fault.

For training the metric and trace detectors, we use the non-anomalous telemetry data as part of the original 2022 AIOps Challenge dataset collected on 03-19, and can be found [here](!https://dataset.aiops-challenge.com/dataset/2022aiops%E6%8C%91%E6%88%98%E8%B5%9B%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE/training_data_normal.tar.gz). 
The [`prep_OpenRCA_Market_dataset`](./prep_OpenRCA_Market_dataset.py) script takes this data, formats it in the same manner as the OpenRCA Market telemetry data, and saves it under `telemetry/normal` for each cloudbed.

### Extraction Details
The ground-truth labelled fault-injection file is [label.csv](../../data/fault-alerts/Online-Boutique-cloudbed-1/label.csv) for cloudbed-1 and [label.csv](../../data/fault-alerts/Online-Boutique-cloudbed-2/label.csv) for cloudbed-2.
For each cloudbed, the extracted alerts can be found in the respective `metrics.json` `traces.json`, and `logs.json` files.

Alerts are produced similarly as described for MicroSS above.

For the metrics, we use the *Oracle KPIs* from OpenRCA. Due to the high number of metric data available in this dataset, we reduce the metric alert space to ensure these do not exceed the model context. The selected KPIs are sufficient to derive the root cause for each injected fault. The *Oracle KPIs* include the following:
```
kpi_Market = {
    "process": ["container_threads",
                ],
    
    "cpu": ["container_cpu_usage_seconds",
            "system.cpu.pct_usage",
            ],
    
    "mem": ["system.mem.used",
            "container_memory_usage_MB",
            ],
    
    "io": ["container_fs_reads./dev/vda",
           "container_fs_writes./dev/vda",
           "system.io.r_s",
           "system.io.w_s",
           "container_fs_writes./dev/vda1",
           "system.disk.used",
           "system.disk.pct_usage",
           ],
    
    "net": ["container_network_receive_packets.eth0",
            "container_network_receive_MB.eth0",
            "recommendationservice-grpc",
            "frontend-http",
            "cartservice-grpc",
            "checkoutservice-grpc",
            "productcatalogservice-grpc",
            "emailservice-grpc",
            "adservice-grpc",
            ],
}
```

#### Note on timestamps
All timestamps are represented as Unix epoch time. 
We therefore present all date-time strings in the UTC timezone. 