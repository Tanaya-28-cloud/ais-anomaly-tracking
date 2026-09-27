# AIS ML Layer

This module implements the standalone ML checkpoint and the optional MQTT
scoring process from `ML_LAYER_BUILD_INSTRUCTIONS.md`.

## Data contract

The source CSVs under `data/terrestrial` and `data/satellite` must contain:

`MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`, and `channel`.

The current datasets contain whole-second and fractional-second timestamps;
the loader accepts both. The label metadata is kept separately under
`data/labels` and is not treated as an AIS source stream.

## Standalone workflow

From `ais-anomaly-tracking`:

```text
python -m ml_layer.data_inspection data
python -m ml_layer.train data models/ais_isolation_forest.joblib --contamination 0.05
python -m ml_layer.evaluate data data/labels/injected_anomalies.csv models/ais_isolation_forest.joblib --report reports/ml_rule_evaluation.json --dark-period-threshold-min 10 --dark-check-interval-min 5 --match-tolerance-min 5
```

The detector is unsupervised. The `--contamination` value is a model decision,
not a measured prevalence of real anomalies; tune it with the project owner
after reviewing the evaluation report.

Each scored record contains the original AIS fields plus `anomaly_score`,
`anomaly_flag`, `anomaly_source`, `ml_output_schema`, and `feature_version`.
The rule engine separately writes `flagged`, `anomaly_type`, and `risk_score`
to `raw_ais_records`; the ML scorer publishes to its own MQTT topic and is not
currently merged into or persisted by that receiver.

## MQTT scoring

Train an artifact first, then run:

```text
python -m ml_layer.mqtt_scorer models/ais_isolation_forest.joblib --broker localhost --output-topic ais/anomalies
```

The scorer subscribes to `ais/terrestrial` and `ais/satellite` and publishes
results to `ais/anomalies`. A scoring error is logged and does not terminate
the subscriber or silently discard the input stream.

## Known integration decisions

- The existing receiver and publisher use the two input topics, so the ML
  process subscribes to both and runs downstream as a separate process. The
  project owner approved this invocation; ML publishes to `ais/anomalies`.
- The rule engine is implemented in `receiver/rules.py`. Its defaults are a
  60-knot implied-speed ceiling, MMSI duplication within 120 seconds and beyond
  5 nautical miles, and a 10-minute dark-period threshold. The offline harness
  defaults to a 15-minute dark threshold and a 5-minute simulated-time sweep;
  evaluation parameters are explicit so both configurations can be assessed.
- Evaluation replays `RuleEngine` chronologically and reports record-level
  ML/rule agreement and disagreement, plus label-window detection counts for
  each anomaly type. Agreement includes the dominant both-unflagged case, so
  inspect `both_flagged`, `ml_only`, and `rule_only` alongside the aggregate.
- The tamper-evident logging schema, dashboard anomaly presentation, latency
  budget, and artifact storage/versioning policy still require project-owner
  confirmation. The DB has `flagged`, `anomaly_type`, and `risk_score` columns,
  but there is no agreed policy yet for combining ML and rule verdicts in those
  fields. The ML output is versioned internally as schema `1.0` and feature
  set `trajectory-v1`.