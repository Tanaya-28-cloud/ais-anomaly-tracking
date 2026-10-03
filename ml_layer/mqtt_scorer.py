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
    detector = scorer.detector
    LOGGER.info(
        "Active ML thresholds: model=%s kernel=rbf nu=%.4f gamma=%s "
        "training_alert_fraction=%.2f%% saved_score_cutoff=%.10g",
        detector.model_name, detector.model.nu, detector.model.gamma,
        detector.alert_fraction * 100, detector.score_threshold,
    )
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
            LOGGER.info(
                "[ML %s] MMSI=%s channel=%s timestamp=%s location=(%s, %s) "
                "score=%.8g cutoff=%.8g",
                "FLAGGED" if result["anomaly_flag"] else "ok",
                result.get("mmsi"), result.get("channel"), result.get("timestamp"),
                result.get("LAT"), result.get("LON"), result["anomaly_score"],
                result["score_threshold"],
            )
            explanation = result.get("shap_explanation")
            if explanation and explanation.get("features"):
                top_features = sorted(
                    explanation["features"],
                    key=lambda item: abs(item["shap_value"]),
                    reverse=True,
                )[:5]
                LOGGER.info(
                    "[SHAP] score=%.8g base=%.8g cutoff=%.8g additivity_residual=%.3g; top contributions: %s",
                    explanation["score"], explanation["base_value"],
                    explanation["score_threshold"], explanation["additivity_residual"],
                    "; ".join(
                        f"{item.get('label', item['name'])}={item['value']:.5g} "
                        f"contribution={item['shap_value']:+.5g} ({item['effect']})"
                        for item in top_features
                    ),
                )
            elif explanation and explanation.get("status") == "error":
                LOGGER.error("[SHAP] %s", explanation.get("message"))
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
