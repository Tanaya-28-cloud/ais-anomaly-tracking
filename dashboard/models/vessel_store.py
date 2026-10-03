"""
Vessel data access layer.

Queries PostgreSQL for each vessel's most recent position. Nothing in
routes/api.py, map.html, or map.js needs to know this changed from
mock data — they only ever talk to get_current_vessels(), never to
the database directly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from receiver import db_writer


def get_current_vessels():
    db_writer.ensure_ml_state_table()
    conn = db_writer.get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (r.mmsi)
                r.mmsi, r.lat, r.lon, r.sog, r.cog, r.channel, r.flagged,
                r.anomaly_type, r.risk_score, r.record_timestamp,
                ml.anomaly_flag, ml.anomaly_score, ml.score_threshold,
                ml.shap_explanation, ml.channel, ml.record_timestamp
            FROM raw_ais_records AS r
            LEFT JOIN LATERAL (
                SELECT state.channel, state.record_timestamp,
                       state.anomaly_flag, state.anomaly_score,
                       state.score_threshold, state.shap_explanation
                FROM ml_vessel_state AS state
                WHERE state.mmsi = r.mmsi
                ORDER BY (state.channel = r.channel) DESC, state.record_timestamp DESC
                LIMIT 1
            ) AS ml ON TRUE
            ORDER BY r.mmsi, r.record_timestamp DESC
        """)
        rows = cur.fetchall()
    return [
        {"mmsi": r[0], "lat": r[1], "lon": r[2], "sog": r[3],
         "cog": r[4], "channel": r[5], "flagged": r[6],
         "rule_anomaly_type": r[7], "rule_risk_score": r[8],
         "timestamp": r[9].isoformat() if r[9] else None,
         "ml_flagged": r[10], "ml_score": r[11], "ml_threshold": r[12],
         "shap_explanation": r[13], "ml_channel": r[14],
         "ml_timestamp": r[15].isoformat() if r[15] else None}
        for r in rows
    ]
