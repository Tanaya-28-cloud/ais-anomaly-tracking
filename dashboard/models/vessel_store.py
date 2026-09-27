"""
Vessel data access layer.

Queries PostgreSQL for each vessel's most recent position. Nothing in
routes/api.py, map.html, or map.js needs to know this changed from
mock data — they only ever talk to get_current_vessels(), never to
the database directly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from receiver import db_writer


def get_current_vessels():
    conn = db_writer.get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (mmsi) mmsi, lat, lon, sog, cog, channel, flagged
            FROM raw_ais_records
            ORDER BY mmsi, record_timestamp DESC
        """)
        rows = cur.fetchall()
    return [
        {"mmsi": r[0], "lat": r[1], "lon": r[2], "sog": r[3],
         "cog": r[4], "channel": r[5], "flagged": r[6]}
        for r in rows
    ]