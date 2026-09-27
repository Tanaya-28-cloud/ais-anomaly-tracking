# AIS ML and Rule Evaluation Summary

Evaluation used the 433,393 terrestrial and 104,585 satellite records, plus 48
injected anomaly scenarios. The Isolation Forest artifact was trained with
`contamination=0.05`; rule replay used the production `RuleEngine` with a
10-minute dark-period threshold, a 5-minute simulated-time sweep, and a
5-minute label matching tolerance.

## Injected Scenario Results

| Anomaly type | ML caught | Rules caught | Both | ML only | Rules only | Neither |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dark period | 6/12 | 5/12 | 3 | 3 | 2 | 4 |
| MMSI duplication | 12/12 | 12/12 | 12 | 0 | 0 | 0 |
| Position jump | 12/12 | 12/12 | 12 | 0 | 0 | 0 |
| Type mismatch | 9/12 | 12/12 | 9 | 0 | 3 | 0 |
| **Total** | **39/48** | **41/48** | **36** | **3** | **5** | **4** |

The ML detector uniquely matched 3 labeled dark-period scenarios that the
rule replay did not match within the configured time tolerance. The rules
uniquely matched 3 type-mismatch and 2 dark-period scenarios. Both systems
matched every labeled MMSI-duplication and position-jump scenario.

## Record-Level Comparison

- ML flagged 26,899 records; rule per-message verdicts flagged 16,331.
- Per-message agreement was 92.37%: 1,103 both flagged, 25,796 ML-only,
  15,228 rules-only, and 495,851 neither flagged.
- The estimated ML false-positive rate was 5.00% among records outside the
  labeled anomaly windows.
- The rule replay also emitted 216 periodic dark-sweep events. Those are
  separate events rather than per-message verdicts.

The high record-level agreement is dominated by records both systems did not
flag; use the scenario-level results and separate ML-only/rules-only counts
for interpretation. The estimated false-positive rate is provisional because
the dataset may contain unlabelled anomalies. ML's Isolation Forest score is
unsupervised and does not establish that ML-only records are true anomalies.

See `ml_rule_evaluation.json` for full metrics. The offline harness uses a
15-minute dark-period threshold, whereas the live `RuleEngine` default is
10 minutes; the harness-default comparison is recorded separately.