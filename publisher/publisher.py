import argparse
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt


def parse_args():
    parser = argparse.ArgumentParser(description="AIS MQTT Publisher")

    parser.add_argument(
        "--file",
        required=True,
        help="Path to AIS CSV file"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="MQTT topic"
    )

    parser.add_argument(
        "--broker",
        default="localhost",
        help="MQTT broker address"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT broker port"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between messages in seconds"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of records to publish; 0 means all"
    )

    return parser.parse_args()


def create_client():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ais-publisher"
    )
    return client


def main():
    args = parse_args()

    print(f"Loading CSV: {args.file}")

    df = pd.read_csv(args.file)

    print(f"Records loaded: {len(df)}")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Publishing first {len(df)} records")

    client = create_client()

    print(f"Connecting to MQTT broker: {args.broker}:{args.port}")

    client.connect(args.broker, args.port, 60)

    client.loop_start()

    print(f"Publishing to topic: {args.topic}")
    print("Press Ctrl+C to stop.\n")

    try:
        for index, row in df.iterrows():

            message = row.to_dict()

            payload = json.dumps(
                message,
                default=str
            )

            result = client.publish(
                args.topic,
                payload,
                qos=0
            )

            result.wait_for_publish()

            print(
                f"[{index + 1}/{len(df)}] "
                f"MMSI={message.get('MMSI')} "
                f"Channel={message.get('channel')}"
            )

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nPublishing stopped by user.")

    finally:
        client.loop_stop()
        client.disconnect()

    print("Publisher finished.")


if __name__ == "__main__":
    main()