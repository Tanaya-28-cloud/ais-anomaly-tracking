"""
Rule-based anomaly detection engine.

Designed to be usable two ways:
  1. Standalone / offline — feed it records from a CSV for testing
     (see scripts/test_rules.py), no MQTT or DB needed.
  2. Live — call engine.process(record) from receiver.py's on_message
     callback once MQTT is wired up. Same code path either way.

A record is expected to be a dict with:
    mmsi (int), lat, lon (float), sog, cog, heading (float),
    timestamp (datetime — already parsed, not a string),
    channel (str), vessel_type (optional)
"""

from math import radians, sin, cos, sqrt, atan2

SPEED_LIMIT_KNOTS = 60          # generous ceiling — tune once you see real data
DUPLICATION_TIME_WINDOW_SEC = 120
DUPLICATION_MIN_DISTANCE_NM = 5
DARK_PERIOD_THRESHOLD_MIN = 10   # default — override via RuleEngine(dark_period_threshold_min=...)


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    R = 3440.065
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def check_speed_jump(current, previous):
    """Flags a position change implying an impossible speed."""
    if previous is None:
        return None
    dt_hours = (current["timestamp"] - previous["timestamp"]).total_seconds() / 3600
    if dt_hours <= 0:
        return None
    dist_nm = haversine_nm(previous["lat"], previous["lon"], current["lat"], current["lon"])
    implied_speed = dist_nm / dt_hours
    if implied_speed > SPEED_LIMIT_KNOTS:
        return {
            "type": "position_jump",
            "severity": min(1.0, implied_speed / 500),
            "observed": {"implied_speed_knots": implied_speed, "distance_nm": dist_nm, "elapsed_minutes": dt_hours * 60},
            "thresholds": {"implied_speed_knots": SPEED_LIMIT_KNOTS},
            "detail": f"Implied speed {implied_speed:.1f} kn > {SPEED_LIMIT_KNOTS} kn over {dt_hours*60:.1f} min",
        }
    return None


def check_mmsi_duplication(current, previous):
    """
    Flags the same MMSI reporting two distant positions within a short
    window — the core signature of MMSI duplication/spoofing.
    Note: this overlaps somewhat with check_speed_jump (both look at
    consecutive records for the same MMSI) but is a distinct semantic
    signal worth keeping separate for explainability.

    NOTE (unchanged from before, flagged for investigation, not fixed
    here): `previous` is the last record for this MMSI regardless of
    channel. If Sudan's injection script wrote the same fabricated
    anomaly into both the terrestrial and satellite CSVs, this rule may
    legitimately fire once per channel for one real injected event —
    that's not necessarily a bug. Use the `channel` field now included
    in test_rules.py's output to inspect this on the real dataset
    before changing this comparison to be channel-aware.
    """
    if previous is None:
        return None
    dt_seconds = abs((current["timestamp"] - previous["timestamp"]).total_seconds())
    if dt_seconds < DUPLICATION_TIME_WINDOW_SEC:
        dist_nm = haversine_nm(previous["lat"], previous["lon"], current["lat"], current["lon"])
        if dist_nm > DUPLICATION_MIN_DISTANCE_NM:
            return {
                "type": "mmsi_duplication",
                "severity": min(1.0, dist_nm / 50),
                "observed": {"distance_nm": dist_nm, "time_difference_seconds": dt_seconds},
                "thresholds": {"distance_nm_minimum": DUPLICATION_MIN_DISTANCE_NM, "time_seconds_maximum": DUPLICATION_TIME_WINDOW_SEC},
                "detail": f"{dist_nm:.1f} nm > {DUPLICATION_MIN_DISTANCE_NM} nm and {dt_seconds:.0f}s < {DUPLICATION_TIME_WINDOW_SEC}s",
            }
    return None


def check_identity_mismatch(current, known_type):
    """Flags a vessel's type field changing partway through its track."""
    current_type = current.get("vessel_type")
    if known_type is not None and current_type is not None and current_type != known_type:
        return {
            "type": "identity_mismatch",
            "severity": 0.7,
            "observed": {"known_type": known_type, "reported_type": current_type},
            "detail": f"Type changed from {known_type} to {current_type}",
        }
    return None


def fuse_results(results):
    """Combines whichever rules triggered into one flagged/score verdict."""
    triggered = [r for r in results if r]
    if not triggered:
        return {"flagged": False, "risk_score": 0.0, "anomaly_types": [], "details": []}
    base = sum(r["severity"] for r in triggered) / len(triggered)
    bonus = 0.1 * (len(triggered) - 1)  # multiple simultaneous rule hits raise confidence
    return {
        "flagged": True,
        "risk_score": round(min(1.0, base + bonus), 3),
        "anomaly_types": [r["type"] for r in triggered],
        "details": triggered,
    }


class RuleEngine:
    """
    Stateful rule engine — keeps track of each vessel's last-seen record,
    first-seen type, last-seen time, and dark-event state, and evaluates
    a new record against that history.

    Dark-period detection is NOT run per-message — see check_dark_vessels()
    below. It's a periodic, time-based sweep, and it's IDEMPOTENT: one
    continuous silence produces exactly one dark_event, tracked via
    self.dark_state, reset the moment the vessel sends a new message.
    """

    def __init__(self, dark_period_threshold_min=DARK_PERIOD_THRESHOLD_MIN):
        self.last_record = {}    # mmsi -> last record dict seen (any channel)
        self.known_type = {}     # mmsi -> vessel type as first observed
        self.last_seen = {}      # mmsi -> timestamp of last message
        self.dark_state = {}     # mmsi -> True while an unresolved dark period is open
        self.latest_sim_time = None  # tracks simulation clock, not wall clock
        self.dark_period_threshold_min = dark_period_threshold_min

    def threshold_summary(self):
        return {
            "position_jump_implied_speed_knots_gt": SPEED_LIMIT_KNOTS,
            "mmsi_duplication_time_seconds_lt": DUPLICATION_TIME_WINDOW_SEC,
            "mmsi_duplication_distance_nm_gt": DUPLICATION_MIN_DISTANCE_NM,
            "dark_period_minutes_gt": self.dark_period_threshold_min,
        }

    def process(self, record):
        mmsi = record["mmsi"]
        previous = self.last_record.get(mmsi)
        known_type = self.known_type.get(mmsi)
        last_seen_time = self.last_seen.get(mmsi)

        results = [
            check_speed_jump(record, previous),
            check_mmsi_duplication(record, previous),
            check_identity_mismatch(record, known_type),
        ]

        # On-arrival dark-gap check. This is deliberately IN ADDITION to
        # the periodic check_dark_vessels() sweep, not a replacement for
        # it — the two catch different cases:
        #   - This check catches a gap that both STARTS and ENDS between
        #     two sweep ticks (confirmed by direct test: such a gap is
        #     otherwise invisible, regardless of threshold or sweep
        #     interval, since no tick ever lands inside the violation
        #     window). Evaluated exactly when the vessel resumes, so it
        #     is immune to the sweep's global-clock phase alignment.
        #   - The periodic sweep still matters for vessels that go dark
        #     and NEVER resume before the dataset/stream ends — there is
        #     no "arrival" for this check to trigger on in that case.
        dark_result = None
        if last_seen_time is not None:
            gap_min = (record["timestamp"] - last_seen_time).total_seconds() / 60
            if gap_min > self.dark_period_threshold_min and not self.dark_state.get(mmsi):
                dark_result = {
                    "type": "dark_event",
                    "severity": min(1.0, gap_min / 120),
                    "observed": {"silence_minutes": gap_min},
                    "thresholds": {"silence_minutes": self.dark_period_threshold_min},
                    "detail": f"No message for {gap_min:.0f} min (detected on resume)",
                }
        results.append(dark_result)

        verdict = fuse_results(results)

        # Update state for next time
        self.last_record[mmsi] = record
        self.last_seen[mmsi] = record["timestamp"]
        if known_type is None and record.get("vessel_type") is not None:
            self.known_type[mmsi] = record["vessel_type"]
        if self.latest_sim_time is None or record["timestamp"] > self.latest_sim_time:
            self.latest_sim_time = record["timestamp"]

        # A message just arrived, so this vessel is no longer dark right
        # now — whether this arrival just fired dark_result (fresh catch)
        # or a prior periodic sweep already reported this same gap
        # (dark_result stays None, avoiding a double report), the state
        # resets here so a FUTURE, independent gap can still be caught.
        self.dark_state[mmsi] = False

        return verdict

    def check_dark_vessels(self, now=None):
        """
        Time-based sweep over all known vessels. Emits exactly ONE
        dark_event per continuous silence period, via self.dark_state —
        calling this repeatedly while a vessel remains dark will NOT
        re-flag it; it only flags the moment silence first crosses the
        threshold, then stays quiet until the vessel resumes (see
        process() above) and a NEW gap later exceeds threshold again.

        Call this periodically based on SIMULATION time (the timestamp
        of the most recently processed message) — see scripts/test_rules.py
        for how the replay loop schedules these calls on a fixed
        simulated-minutes interval, not a record count and not
        real wall-clock time.
        """
        if now is None:
            now = self.latest_sim_time
        if now is None:
            return []

        newly_flagged = []
        for mmsi, last_time in self.last_seen.items():
            gap_min = (now - last_time).total_seconds() / 60
            if gap_min > self.dark_period_threshold_min and not self.dark_state.get(mmsi):
                self.dark_state[mmsi] = True
                newly_flagged.append({
                    "mmsi": mmsi,
                    "type": "dark_event",
                    "severity": min(1.0, gap_min / 120),
                    "lat": self.last_record[mmsi].get("lat"),
                    "lon": self.last_record[mmsi].get("lon"),
                    "observed": {"silence_minutes": gap_min},
                    "thresholds": {"silence_minutes": self.dark_period_threshold_min},
                    "detail": f"No message for {gap_min:.0f} min",
                    "timestamp": now,
                })
        return newly_flagged
