"""Subscribe to AIS MQTT topics and publish ML anomaly results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import paho.mqtt.client as mqtt

from .model import load_detector
from .streaming import StreamingScorer


LOGGER = logging.getLogger("ml_layer.mqtt_scorer")
INPUT_TOPICS = ("ais/terrestrial", "ais/satellite")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--output-topic", default="ais/anomalies")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scorer = StreamingScorer(load_detector(args.artifact))
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ais-ml-scorer",
    )

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            for topic in INPUT_TOPICS:
                client.subscribe(topic)
            LOGGER.info("subscribed to %s", ", ".join(INPUT_TOPICS))
        else:
            LOGGER.error("MQTT connection failed: %s", reason_code)

    def on_message(client, userdata, message):
        try:
            record = json.loads(message.payload.decode("utf-8"))
            result = scorer.score_record(record)
            client.publish(args.output_topic, json.dumps(result), qos=0)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            LOGGER.exception("invalid AIS record on topic %s", message.topic)
        except Exception:
            LOGGER.exception("ML scoring failed for topic %s", message.topic)

    client.on_connect = on_connect
    client.on_message = on_message
    LOGGER.info("connecting to MQTT broker %s:%s", args.broker, args.port)
    client.connect(args.broker, args.port, 60)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        LOGGER.info("ML scorer stopped")
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
