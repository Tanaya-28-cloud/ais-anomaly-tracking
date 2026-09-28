"""
Vessel-side replay publisher.

Reads the terrestrial AND satellite CSVs, merges them into ONE
chronological stream (by BaseDateTime), and publishes each row to
ais/terrestrial or ais/satellite according to the file it came from.

Why not run publisher.py twice? Two publisher.py processes both use
client_id "ais-publisher" (the broker disconnects the older one), and
publishing the files one after the other feeds the rule engine and the ML
scorer out-of-order timestamps. This script avoids both problems.

Put in publisher/replay_publisher.py. Run from the repository root:
    python publisher/replay_publisher.py --broker 192.168.1.20 \
        --terrestrial data/terrestrial/cleaned_ais_data.csv \
        --satellite   data/satellite/<your_satellite_file>.csv --limit 200

Only needs: pip install paho-mqtt pandas
"""

import argparse
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt


def load(path, channel):
    df = pd.read_csv(path)
    df["_ts"] = pd.to_datetime(df["BaseDateTime"], format="mixed", utc=True)
    df["_topic"] = f"ais/{channel}"
    if "channel" not in df.columns:
        df["channel"] = channel
    return df


def main():
    p = argparse.ArgumentParser(description="Merged chronological AIS replay over MQTT")
    p.add_argument("--broker", required=True, help="IP of the machine running Mosquitto")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--terrestrial", required=True)
    p.add_argument("--satellite", required=True)
    p.add_argument("--delay", type=float, default=0.1, help="seconds between messages")
    p.add_argument("--limit", type=int, default=0, help="publish only the first N merged rows (0 = all)")
    p.add_argument("--mmsi", type=int, nargs="+", help="only publish these MMSIs (e.g. one injected-anomaly vessel)")
    p.add_argument("--client-id", default="ais-vessel-replay")
    args = p.parse_args()

    df = pd.concat([load(args.terrestrial, "terrestrial"), load(args.satellite, "satellite")],
                   ignore_index=True)
    if args.mmsi:
        df = df[df["MMSI"].isin(args.mmsi)]
    df = df.sort_values(["_ts", "MMSI", "_topic"], kind="stable").reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"Publishing {len(df)} merged rows "
          f"({(df['_topic'] == 'ais/terrestrial').sum()} terrestrial, "
          f"{(df['_topic'] == 'ais/satellite').sum()} satellite)")

    topics = df["_topic"].tolist()
    payload_df = df.drop(columns=["_ts", "_topic"]).astype(object)
    payload_df = payload_df.where(payload_df.notna(), None)   # NaN -> null (valid JSON)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id)
    client.connect(args.broker, args.port, 60)
    client.loop_start()
    try:
        for i, (topic, row) in enumerate(zip(topics, payload_df.to_dict("records")), 1):
            client.publish(topic, json.dumps(row, default=str), qos=0).wait_for_publish()
            print(f"[{i}/{len(df)}] {topic}  MMSI={row['MMSI']}  t={row['BaseDateTime']}")
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
    print("Replay finished.")


if __name__ == "__main__":
    main()
