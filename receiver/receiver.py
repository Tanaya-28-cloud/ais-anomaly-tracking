"""
Hybrid AIS receiver (rule engine) -- subscribes to ais/terrestrial and
ais/satellite, runs every message through the RuleEngine, prints the
verdict WITH the reason on the terminal, and (optionally) writes to
PostgreSQL.

Replaces receiver/receiver.py. Changes vs. the original:
  * --no-db      : run without PostgreSQL (terminal-only test mode)
  * prints the reason text for every flag (verdict["details"])
  * blank/NaN VesselType is treated as "unknown" (original code compared
    NaN != NaN and raised a false identity_mismatch on every message)
  * one bad message no longer kills the MQTT loop
  * periodic dark-vessel sweep on DATA time (the original receiver never
    called check_dark_vessels(), so a vessel that goes silent and never
    resumes was never flagged live)
  * summary printed on Ctrl+C

Run from the repository ROOT (not from inside receiver/):
    python -m receiver.receiver --broker 127.0.0.1 --no-db
"""

import argparse
import json
import math
from collections import Counter

import pandas as pd
import paho.mqtt.client as mqtt

from receiver.rules import RuleEngine

engine = RuleEngine()
USE_DB = True
DARK_SWEEP_MIN = 5.0
_last_sweep_time = None
stats = Counter()
db_writer = None  # imported lazily so --no-db needs no psycopg2


def _num(value):
    """float(value), or None for None / '' / 'nan' / NaN."""
    if value is None or value == "" or value == "nan":
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _text_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none"):
        return None
    return value


def parse_record(raw: dict) -> dict:
    """Normalise a raw MQTT payload (original CSV column names) for RuleEngine/db_writer."""
    return {
        "mmsi": int(raw["MMSI"]),
        "lat": float(raw["LAT"]),
        "lon": float(raw["LON"]),
        "sog": _num(raw.get("SOG")),
        "cog": _num(raw.get("COG")),
        "heading": _num(raw.get("Heading")),
        "vessel_type": _text_or_none(raw.get("VesselType")),
        "timestamp": pd.to_datetime(raw["BaseDateTime"]),
        "channel": raw.get("channel", "terrestrial"),
    }


def _maybe_sweep(now):
    """Run check_dark_vessels every DARK_SWEEP_MIN minutes of DATA time."""
    global _last_sweep_time
    if DARK_SWEEP_MIN <= 0:
        return
    if _last_sweep_time is None:
        _last_sweep_time = now
        return
    if (now - _last_sweep_time).total_seconds() / 60 >= DARK_SWEEP_MIN:
        for hit in engine.check_dark_vessels(now=now):
            stats["dark_sweep"] += 1
            print(f"[DARK-SWEEP] MMSI {hit['mmsi']} @ {hit['timestamp']} "
                  f"(score {min(1.0, hit['severity']):.3f})\n    -> dark_event: {hit['detail']}")
        _last_sweep_time = now


def handle_message(topic: str, payload: bytes):
    raw = json.loads(payload.decode("utf-8"))
    record = parse_record(raw)

    _maybe_sweep(record["timestamp"])
    verdict = engine.process(record)
    stats["total"] += 1

    tag = f"MMSI {record['mmsi']} [{record['channel']}] @ {record['timestamp']}"
    if verdict["flagged"]:
        stats["flagged"] += 1
        for t in verdict["anomaly_types"]:
            stats[f"type:{t}"] += 1
        print(f"[FLAGGED] {tag}  score={verdict['risk_score']}  "
              f"pos=({record['lat']:.4f}, {record['lon']:.4f})")
        for d in verdict["details"]:
            print(f"    -> {d['type']}: {d['detail']}  (severity {d['severity']:.2f})")
    else:
        print(f"[ok]      {tag}")

    if USE_DB:
        db_writer.insert_record(record)
        if verdict["flagged"]:
            db_writer.update_flag(
                mmsi=record["mmsi"],
                record_timestamp=record["timestamp"],
                flagged=True,
                anomaly_type=",".join(verdict["anomaly_types"]),
                risk_score=verdict["risk_score"],
            )


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to broker (reason_code={reason_code}) - subscribing to both channels")
    client.subscribe("ais/terrestrial")
    client.subscribe("ais/satellite")


def on_message(client, userdata, msg):
    try:
        handle_message(msg.topic, msg.payload)
    except Exception as exc:  # keep the loop alive; show what went wrong
        stats["errors"] += 1
        print(f"[ERROR] could not process message on {msg.topic}: {exc!r}")


def print_summary():
    print("\n===== receiver summary =====")
    print(f"messages processed : {stats['total']}")
    print(f"flagged            : {stats['flagged']}")
    for k in sorted(k for k in stats if k.startswith("type:")):
        print(f"  {k[5:]:<18}: {stats[k]}")
    print(f"dark-sweep events  : {stats['dark_sweep']}")
    print(f"errors             : {stats['errors']}")


def main():
    global USE_DB, DARK_SWEEP_MIN, db_writer, engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", required=True, help="IP/hostname of the MQTT broker")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--no-db", action="store_true", help="terminal-only mode, no PostgreSQL")
    parser.add_argument("--dark-threshold-min", type=float, default=10.0)
    parser.add_argument("--dark-sweep-min", type=float, default=5.0,
                        help="dark sweep interval in DATA minutes; 0 disables")
    args = parser.parse_args()

    USE_DB = not args.no_db
    DARK_SWEEP_MIN = args.dark_sweep_min
    engine = RuleEngine(dark_period_threshold_min=args.dark_threshold_min)
    if USE_DB:
        from receiver import db_writer as _db
        db_writer = _db

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ais-receiver")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, 60)
    mode = "writing to PostgreSQL" if USE_DB else "terminal only (no DB)"
    print(f"Rule receiver running - {mode}. Ctrl+C to stop.")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        print_summary()


if __name__ == "__main__":
    main()