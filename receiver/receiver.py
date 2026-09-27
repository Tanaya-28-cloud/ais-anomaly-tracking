"""
Hybrid AIS receiver — subscribes to both the terrestrial and satellite
MQTT topics at once, fuses both channels, runs every incoming message
through the rule engine, and stores the result in PostgreSQL.

Matched to Tanmay's publisher.py, which sends the full raw CSV row
(original column names: MMSI, LAT, LON, BaseDateTime, VesselType, etc.)
rather than a curated lowercase field set, and uses the paho-mqtt v2
CallbackAPIVersion.VERSION2 API.

Usage:
    python receiver/receiver.py --broker 192.168.1.5

Requires: pip install paho-mqtt psycopg2-binary pandas
"""

import argparse
import json

import pandas as pd
import paho.mqtt.client as mqtt

from receiver.rules import RuleEngine
from receiver import db_writer

# One shared engine instance for the whole process. It's stateful
# (tracks last-seen position/type per vessel across calls), so it has
# to persist for the life of the program, not be recreated per message.
engine = RuleEngine()


def parse_record(raw: dict) -> dict:
    """
    Normalizes a raw MQTT payload (the original CSV column names,
    values possibly stringified by json.dumps(default=str)) into the
    lowercase, correctly-typed shape RuleEngine and db_writer expect.
    Explicit casts here are deliberate — values may arrive as either
    native numbers or strings depending on how pandas/numpy types got
    serialized, so we don't trust the wire type either way.
    """
    return {
        "mmsi": int(raw["MMSI"]),
        "lat": float(raw["LAT"]),
        "lon": float(raw["LON"]),
        "sog": float(raw["SOG"]) if raw.get("SOG") not in (None, "", "nan") else None,
        "cog": float(raw["COG"]) if raw.get("COG") not in (None, "", "nan") else None,
        "heading": float(raw["Heading"]) if raw.get("Heading") not in (None, "", "nan") else None,
        "vessel_type": raw.get("VesselType"),
        # pd.to_datetime handles both ISO ('...T...') and pandas'
        # default str(Timestamp) space-separated format, so this is
        # safe regardless of exactly how BaseDateTime got stringified.
        "timestamp": pd.to_datetime(raw["BaseDateTime"]),
        "channel": raw.get("channel", "terrestrial"),
    }


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to broker (reason_code={reason_code}) — subscribing to both channels")
    client.subscribe("ais/terrestrial")
    client.subscribe("ais/satellite")


def on_message(client, userdata, msg):
    raw = json.loads(msg.payload.decode())
    record = parse_record(raw)

    verdict = engine.process(record)

    if verdict["flagged"]:
        print(f"[FLAGGED] MMSI {record['mmsi']} [{record['channel']}] @ {record['timestamp']} "
              f"— {verdict['anomaly_types']} (score {verdict['risk_score']})")
    else:
        print(f"[ok] MMSI {record['mmsi']} [{record['channel']}] @ {record['timestamp']}")

    # Write the raw record first (flagged defaults to FALSE per the
    # schema), then only make a second call if it actually needs
    # updating — keeps the common (non-flagged) case to one query.
    db_writer.insert_record(record)

    if verdict["flagged"]:
        db_writer.update_flag(
            mmsi=record["mmsi"],
            record_timestamp=record["timestamp"],
            flagged=True,
            anomaly_type=",".join(verdict["anomaly_types"]),
            risk_score=verdict["risk_score"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", required=True, help="IP address of the MQTT broker laptop")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ais-receiver")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.broker, args.port, 60)
    print("Listening on both channels, running the rule engine, writing to DB... (Ctrl+C to stop)")
    client.loop_forever()


if __name__ == "__main__":
    main()