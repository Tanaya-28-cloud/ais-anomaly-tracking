# AIS Anomaly Tracking

A hybrid AIS anomaly tracking system that simulates terrestrial and satellite AIS data transmission using MQTT.

## Current Progress

The current implementation focuses on the AIS data transmission environment.

### Completed

- Python 3.11 virtual environment setup
- Mosquitto MQTT broker setup
- MQTT publisher implementation
- MQTT receiver implementation
- Terrestrial AIS data publishing
- Simulated satellite AIS data publishing
- Dual MQTT topics:
  - `ais/terrestrial`
  - `ais/satellite`
- JSON-based AIS message transmission
- Local hybrid receiver testing
- GitHub repository setup

## System Architecture

```text
                 AIS Data Sources
                       |
              +--------+--------+
              |                 |
              v                 v
       Terrestrial CSV    Satellite CSV
              |                 |
              v                 v
        publisher.py      publisher.py
              |                 |
              v                 v
        ais/terrestrial   ais/satellite
              |                 |
              +--------+--------+
                       |
                       v
                MQTT Broker
                 (Mosquitto)
                       |
                       v
                  receiver.py
                       |
                       v
                Hybrid AIS Stream