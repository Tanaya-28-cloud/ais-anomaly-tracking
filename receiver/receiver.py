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
import os
import uuid
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


def print_thresholds():
    thresholds = engine.threshold_summary()
    print("Active receiver thresholds:")
    print(f"  position jump: implied speed > {thresholds['position_jump_implied_speed_knots_gt']} kn")
    print("  MMSI duplication: same MMSI, time difference < "
          f"{thresholds['mmsi_duplication_time_seconds_lt']} s AND distance > "
          f"{thresholds['mmsi_duplication_distance_nm_gt']} nm")
    print(f"  dark period: silence > {thresholds['dark_period_minutes_gt']} min; sweep every {DARK_SWEEP_MIN:g} data-time min")
    print("  identity/type mismatch: reported non-null type differs from first known non-null type (categorical; no numeric cutoff)")


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
                  f"location=({hit.get('lat')}, {hit.get('lon')}) "
                  f"threshold=silence>{engine.dark_period_threshold_min:g}min "
                  f"(score {min(1.0, hit['severity']):.3f})\n    -> dark_event: {hit['detail']}")
        _last_sweep_time = now


def handle_message(topic: str, payload: bytes):
    raw = json.loads(payload.decode("utf-8"))
    record = parse_record(raw)

    _maybe_sweep(record["timestamp"])
    verdict = engine.process(record)
    stats["total"] += 1

    tag = f"MMSI {record['mmsi']} [{record['channel']}] @ {record['timestamp']}"
    location = f"location=({record['lat']:.6f}, {record['lon']:.6f})"
    thresholds = engine.threshold_summary()
    threshold_text = (
        f"thresholds=[jump_speed>{thresholds['position_jump_implied_speed_knots_gt']}kn; "
        f"duplicate_dt<{thresholds['mmsi_duplication_time_seconds_lt']}s AND "
        f"distance>{thresholds['mmsi_duplication_distance_nm_gt']}nm; "
        f"dark_gap>{thresholds['dark_period_minutes_gt']:g}min; "
        "identity_type_change=reported_nonnull!=first_known_nonnull]"
    )
    if verdict["flagged"]:
        stats["flagged"] += 1
        for t in verdict["anomaly_types"]:
            stats[f"type:{t}"] += 1
        print(f"[FLAGGED] {tag}  {location}  {threshold_text}  rule_score={verdict['risk_score']}")
        for d in verdict["details"]:
            print(f"    -> {d['type']}: {d['detail']}  (severity {d['severity']:.2f}; "
                  f"thresholds={d.get('thresholds', 'categorical comparison')})")
    else:
        print(f"[ok]      {tag}  {location}  {threshold_text}")

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


def handle_ml_message(payload: bytes):
    """Store the ML stream's latest result and print alert explanation details."""
    result = json.loads(payload.decode("utf-8"))
    if USE_DB:
        db_writer.upsert_ml_state(result)
    if not result.get("anomaly_flag"):
        return
    print(
        f"[ML-FLAGGED] MMSI {result.get('mmsi')} [{result.get('channel')}] "
        f"@ {result.get('timestamp')} "
        f"location=({result.get('LAT')}, {result.get('LON')}) "
        f"score={result.get('anomaly_score'):.8g} "
        f"cutoff={result.get('score_threshold'):.8g}"
    )
    explanation = result.get("shap_explanation")
    if explanation and explanation.get("features"):
        top = sorted(explanation.get("features", []),
                     key=lambda item: abs(item.get("shap_value", 0)), reverse=True)[:5]
        print(
            f"    SHAP score={explanation.get('score'):.8g}, "
            f"base={explanation.get('base_value'):.8g}, "
            f"cutoff={explanation.get('score_threshold'):.8g}, "
            f"additivity_residual={explanation.get('additivity_residual'):.3g}"
        )
        for item in top:
            print(
                f"      {item.get('label', item['name'])}={item['value']:.5g}: "
                f"{item['shap_value']:+.5g} ({item['effect']})"
            )
    elif explanation and explanation.get("status") == "error":
        print(f"    SHAP explanation unavailable: {explanation.get('message')}")


connect_count = 0


def on_connect(client, userdata, flags, reason_code, properties=None):
    """Subscribe only. No reconnect / no new client in here."""
    global connect_count
    connect_count += 1
    if getattr(reason_code, "is_failure", reason_code != 0):
        print(f"[MQTT] connection REFUSED by broker: {reason_code}")
        return
    print(f"Connected to broker (reason_code={reason_code}, connection #{connect_count}) "
          f"as client_id={userdata}")
    if connect_count > 1:
        print("[MQTT] WARNING: this is a RE-connection - see the [MQTT] DISCONNECTED line above for the cause")
    client.subscribe("ais/terrestrial")
    client.subscribe("ais/satellite")
    client.subscribe("ais/anomalies")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    """Prints WHY the connection dropped. paho (loop_forever) then re-connects by itself."""
    print(f"[MQTT] DISCONNECTED: reason_code={reason_code} (value={getattr(reason_code, 'value', reason_code)}) "
          f"- 0 = we disconnected on purpose; 142 'Session taken over' = another client used the SAME client_id; "
          f"other/unspecified = network or broker closed the socket")


def on_subscribe(client, userdata, mid, reason_codes, properties=None):
    print(f"[MQTT] subscribed OK: {[str(r) for r in reason_codes]}")


def on_message(client, userdata, msg):
    try:
        if msg.topic == "ais/anomalies":
            handle_ml_message(msg.payload)
        else:
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
    parser.add_argument("--client-id", default=None, help="override the auto-generated unique MQTT client id")
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

    # Unique id per process. A FIXED id ("ais-receiver") is what causes the
    # connect/disconnect ping-pong when two processes (or two laptops) use it:
    # the broker closes the older session every time the other one connects.
    client_id = args.client_id or f"ais-receiver-{os.getpid()}-{uuid.uuid4().hex[:4]}"
    # MQTT v5 so the broker can tell us the reason ("Session taken over").
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                         protocol=mqtt.MQTTv5)
    client.user_data_set(client_id)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"Connecting to {args.broker}:{args.port} as {client_id} ...")
    print_thresholds()
    try:
        client.connect(args.broker, args.port, keepalive=60)   # called exactly once
    except OSError as exc:
        print(f"[MQTT] cannot reach broker {args.broker}:{args.port}: {exc}")
        return
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
