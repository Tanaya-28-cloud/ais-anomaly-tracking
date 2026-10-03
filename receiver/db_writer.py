"""
Writes incoming AIS records into PostgreSQL as they arrive.

Import and call insert_record() from receiver.py's on_message callback,
once per parsed message. A single persistent connection is kept open
and reused (opening a new connection per message would be far too slow
for a real-time stream).

Requires: pip install psycopg2-binary
"""

import os
import json
import psycopg2
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

_conn = None
_ml_table_ready = False


def get_connection():
    """Returns a persistent DB connection, opening one on first call."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            dbname=os.environ.get("DB_NAME", "ais_project"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", ""),
        )
    return _conn


def ensure_ml_state_table():
    """Create the ML state table on older databases without requiring a reset."""
    global _ml_table_ready
    if _ml_table_ready:
        return
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ml_vessel_state (
                mmsi BIGINT NOT NULL,
                channel VARCHAR(20) NOT NULL,
                record_timestamp TIMESTAMP NOT NULL,
                lat DOUBLE PRECISION NOT NULL,
                lon DOUBLE PRECISION NOT NULL,
                anomaly_score DOUBLE PRECISION NOT NULL,
                score_threshold DOUBLE PRECISION NOT NULL,
                anomaly_flag BOOLEAN NOT NULL,
                shap_explanation JSONB,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (mmsi, channel)
            )
        """)
    conn.commit()
    _ml_table_ready = True


def upsert_ml_state(result: dict):
    """Persist the latest per-channel ML score and its local explanation."""
    ensure_ml_state_table()
    conn = get_connection()
    timestamp = result.get("timestamp") or result.get("BaseDateTime")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    explanation = result.get("shap_explanation")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_vessel_state
                (mmsi, channel, record_timestamp, lat, lon, anomaly_score,
                 score_threshold, anomaly_flag, shap_explanation, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (mmsi, channel) DO UPDATE SET
                record_timestamp = EXCLUDED.record_timestamp,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                anomaly_score = EXCLUDED.anomaly_score,
                score_threshold = EXCLUDED.score_threshold,
                anomaly_flag = EXCLUDED.anomaly_flag,
                shap_explanation = EXCLUDED.shap_explanation,
                updated_at = NOW()
            WHERE EXCLUDED.record_timestamp >= ml_vessel_state.record_timestamp
        """, (
            int(result.get("mmsi", result.get("MMSI"))),
            str(result.get("channel") or "unknown"),
            timestamp,
            float(result.get("LAT", result.get("lat"))),
            float(result.get("LON", result.get("lon"))),
            float(result["anomaly_score"]),
            float(result["score_threshold"]),
            bool(result["anomaly_flag"]),
            json.dumps(explanation) if explanation is not None else None,
        ))
    conn.commit()


def _as_datetime(ts):
    """
    Accepts either an already-parsed datetime/pandas.Timestamp object
    (what receiver.py's parse_record() and populate_db.py's
    load_records() actually produce) or a raw ISO string, and returns
    a proper datetime either way. Without this, insert_record() would
    crash on real data — pd.to_datetime() upstream means timestamp is
    no longer a string by the time it gets here.
    """
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    return ts


def insert_record(record: dict):
    """
    Inserts one parsed AIS message into raw_ais_records.

    record is expected to have: mmsi, lat, lon, sog, cog, heading,
    timestamp (datetime/Timestamp object OR ISO string — both handled),
    channel.
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw_ais_records
                (mmsi, lat, lon, sog, cog, heading, channel, record_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record["mmsi"],
                record["lat"],
                record["lon"],
                record.get("sog"),
                record.get("cog"),
                record.get("heading"),
                record["channel"],
                _as_datetime(record["timestamp"]),
            ),
        )
    conn.commit()


def update_flag(mmsi: int, record_timestamp: datetime, flagged: bool,
                 anomaly_type: str = None, risk_score: float = None):
    """
    Called by the rule engine (next phase) after it scores a record —
    updates the same row rather than writing a separate table, so the
    dashboard's live map query stays simple (one table, one row per
    message, flagged status already attached).
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE raw_ais_records
            SET flagged = %s, anomaly_type = %s, risk_score = %s
            WHERE mmsi = %s AND record_timestamp = %s
            """,
            (flagged, anomaly_type, risk_score, mmsi, record_timestamp),
        )
    conn.commit()


def get_recent_history(mmsi: int, limit: int = 30):
    """
    Returns the last `limit` records for a given vessel, most recent
    first — this is the sliding-window query the feature engineering
    step (implied speed, heading divergence, etc.) will call.
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT mmsi, lat, lon, sog, cog, heading, channel, record_timestamp
            FROM raw_ais_records
            WHERE mmsi = %s
            ORDER BY record_timestamp DESC
            LIMIT %s
            """,
            (mmsi, limit),
        )
        return cur.fetchall()
