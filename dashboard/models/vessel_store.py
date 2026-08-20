"""
Vessel data access layer.

Right now this returns mock data so the dashboard can be built and
tested independently of the streaming pipeline (which your teammate
is building in parallel). Once the receiver/database exist, replace
the body of get_current_vessels() with a real query — nothing in
routes/api.py, map.html, or map.js needs to change, since they only
ever talk to this function, not to the data source directly.

Example of what the real version will look like later:

    import psycopg2

    def get_current_vessels():
        conn = psycopg2.connect(...)
        cur = conn.cursor()
        cur.execute('''
            SELECT DISTINCT ON (mmsi) mmsi, lat, lon, sog, cog, channel, flagged
            FROM raw_ais_records
            ORDER BY mmsi, timestamp DESC
        ''')
        rows = cur.fetchall()
        return [
            {"mmsi": r[0], "lat": r[1], "lon": r[2], "sog": r[3],
             "cog": r[4], "channel": r[5], "flagged": r[6]}
            for r in rows
        ]
"""


def get_current_vessels():
    # TODO(integration): replace with a real DB query or a shared
    # in-memory buffer read from receiver.py once that's ready.
    return [
        {"mmsi": 366123456, "lat": 33.7458, "lon": -118.2601, "sog": 12.4,
         "cog": 271.5, "channel": "terrestrial", "flagged": False},
        {"mmsi": 366123457, "lat": 33.7500, "lon": -118.2700, "sog": 8.1,
         "cog": 90.0, "channel": "satellite", "flagged": True},
        {"mmsi": 366123458, "lat": 33.7400, "lon": -118.2550, "sog": 15.2,
         "cog": 180.0, "channel": "terrestrial", "flagged": False},
    ]
