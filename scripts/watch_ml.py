"""
Terminal viewer for the ML scorer's output topic (ais/anomalies).

Put in scripts/watch_ml.py. Run from anywhere:
    python scripts/watch_ml.py --broker 127.0.0.1            # ML-flagged only
    python scripts/watch_ml.py --broker 127.0.0.1 --all      # every scored message

RBF One-Class SVM gives a score and a flag, not a reason. The "hints" below
are simple threshold read-outs of the model's input features so you can see
WHAT looked unusual -- they are not model-derived explanations.
"""

import argparse
import json

import paho.mqtt.client as mqtt


def hints(f):
    out = []
    if not f:
        return out
    v = f.get("implied_speed_knots")
    if v is not None and v > 60:
        out.append(f"implied speed {v:.0f} kn")
    gap = f.get("time_since_last_transmission_seconds")
    if gap is not None and gap > 600:
        out.append(f"silent for {gap/60:.0f} min before this message")
    d = f.get("distance_from_last_km")
    if d is not None and d > 20:
        out.append(f"moved {d:.1f} km since previous report")
    s = f.get("sog_delta")
    if s is not None and s > 10:
        out.append(f"speed changed by {s:.1f} kn")
    c = f.get("cog_delta")
    if c is not None and c > 90:
        out.append(f"course changed by {c:.0f} deg")
    if f.get("non_positive_time_delta") == 1:
        out.append("timestamp not after previous report (out-of-order/duplicate)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default="ais/anomalies")
    p.add_argument("--all", action="store_true", help="print unflagged messages too")
    a = p.parse_args()
    seen = {"total": 0, "flagged": 0}

    def on_connect(c, u, fl, rc, props=None):
        print(f"Connected (reason_code={rc}); watching {a.topic}")
        c.subscribe(a.topic)

    def on_message(c, u, m):
        try:
            r = json.loads(m.payload.decode())
            seen["total"] += 1
            tag = f"MMSI {r.get('mmsi')} [{r.get('channel')}] @ {r.get('timestamp')}"
            if r.get("anomaly_flag"):
                seen["flagged"] += 1
                print(f"[ML-FLAGGED] {tag}  score={r['anomaly_score']:.3f}")
                for h in hints(r.get("ml_features")):
                    print(f"    hint: {h}")
            elif a.all:
                print(f"[ml-ok]      {tag}  score={r['anomaly_score']:.3f}")
        except Exception as e:
            print(f"[ERROR] {e!r}")

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ais-ml-watcher")
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(a.broker, a.port, 60)
    try:
        c.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        c.disconnect()
        print(f"\nML scored {seen['total']} messages, flagged {seen['flagged']}")


if __name__ == "__main__":
    main()
