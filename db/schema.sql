-- AIS Project — Database Schema
-- Run this once to set up the database: psql -U postgres -d ais_project -f schema.sql

CREATE TABLE IF NOT EXISTS raw_ais_records (
    id              SERIAL PRIMARY KEY,
    mmsi            BIGINT NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    sog             DOUBLE PRECISION,
    cog             DOUBLE PRECISION,
    heading         DOUBLE PRECISION,
    channel         VARCHAR(20) NOT NULL,           -- 'terrestrial' or 'satellite'
    record_timestamp TIMESTAMP NOT NULL,             -- the AIS message's own timestamp
    received_at     TIMESTAMP NOT NULL DEFAULT NOW(), -- when our system actually stored it

    -- Filled in later by the rule engine / ML layer — nullable for now,
    -- since Phase 1 just writes raw records. Phase 2's encrypted
    -- anomaly_log table is separate and only holds flagged events.
    flagged         BOOLEAN DEFAULT FALSE,
    anomaly_type    VARCHAR(50),                     -- e.g. 'mmsi_duplication', 'position_jump'
    risk_score      DOUBLE PRECISION
);

-- Speeds up the two query patterns everything downstream needs:
-- (1) "give me vessel X's recent history" — sliding-window feature engineering
-- (2) "give me the latest position per vessel" — dashboard's live map
CREATE INDEX IF NOT EXISTS idx_mmsi_timestamp ON raw_ais_records (mmsi, record_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_flagged ON raw_ais_records (flagged) WHERE flagged = TRUE;

-- Latest ML score/explanation per vessel and AIS input channel. Keeping this
-- separate from raw_ais_records preserves the independent rule and ML verdicts.
CREATE TABLE IF NOT EXISTS ml_vessel_state (
    mmsi             BIGINT NOT NULL,
    channel          VARCHAR(20) NOT NULL,
    record_timestamp TIMESTAMP NOT NULL,
    lat              DOUBLE PRECISION NOT NULL,
    lon              DOUBLE PRECISION NOT NULL,
    anomaly_score    DOUBLE PRECISION NOT NULL,
    score_threshold  DOUBLE PRECISION NOT NULL,
    anomaly_flag     BOOLEAN NOT NULL,
    shap_explanation JSONB,
    updated_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (mmsi, channel)
);
CREATE INDEX IF NOT EXISTS idx_ml_vessel_state_timestamp
    ON ml_vessel_state (record_timestamp DESC);
