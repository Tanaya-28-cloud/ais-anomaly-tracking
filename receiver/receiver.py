import argparse
import json

import paho.mqtt.client as mqtt


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid AIS MQTT Receiver")

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

    return parser.parse_args()


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT broker")
        print("Subscribing to:")
        print("  - ais/terrestrial")
        print("  - ais/satellite")

        client.subscribe("ais/terrestrial")
        client.subscribe("ais/satellite")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode("utf-8"))

        print(
            f"[{data.get('channel', 'unknown').upper()}] "
            f"MMSI={data.get('MMSI')} "
            f"LAT={data.get('LAT')} "
            f"LON={data.get('LON')} "
            f"SOG={data.get('SOG')} "
            f"COG={data.get('COG')} "
            f"TIME={data.get('BaseDateTime')}"
        )

    except json.JSONDecodeError:
        print("Invalid JSON received")

    except Exception as e:
        print(f"Error processing message: {e}")


def main():
    args = parse_args()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="hybrid-ais-receiver"
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker: {args.broker}:{args.port}")

    client.connect(args.broker, args.port, 60)

    print("Receiver started. Waiting for AIS messages...")
    print("Press Ctrl+C to stop.\n")

    try:
        client.loop_forever()

    except KeyboardInterrupt:
        print("\nReceiver stopped.")

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()