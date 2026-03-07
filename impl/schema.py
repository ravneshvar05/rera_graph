"""
schema.py — Creates all SQLite tables and FTS5 virtual table.

Tables:
  projects          — one row per brochure/project
  project_landmarks — one row per nearby landmark (normalized)
  project_amenities — one row per amenity string (normalized)
  units             — one row per unit variant (with derived boolean flags)
  rooms             — one row per room in every unit (zero data loss)

FTS5 virtual table:
  projects_fts      — full-text search across all text fields

Run this directly to recreate schema on an empty DB, or call create_all_tables()
from ingest.py.
"""

import sqlite3
from config import settings
from logger import logger


DDL = """
-- ─── Enable WAL mode for better concurrent reads ─────────────────────────────
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

-- ─── projects ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projects (
    project_id          TEXT PRIMARY KEY,
    project_name        TEXT,
    developer_name      TEXT,
    rera_number         TEXT,
    project_status      TEXT,
    possession_date     TEXT,

    -- location
    city                TEXT,
    neighbourhood       TEXT,
    address             TEXT,
    pin_code            TEXT,

    -- society layout
    has_clubhouse       INTEGER DEFAULT 0,
    has_pool            INTEGER DEFAULT 0,
    has_park            INTEGER DEFAULT 0,
    has_sports_courts   INTEGER DEFAULT 0,
    has_parking         INTEGER DEFAULT 0,
    has_commercial_shops INTEGER DEFAULT 0,
    total_buildings     INTEGER,
    total_villas        INTEGER,
    society_description TEXT,
    road_widths         TEXT,           -- joined string

    -- meta
    source_file         TEXT,
    ingested_at         TEXT
);

-- ─── project_landmarks ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS project_landmarks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    landmark_name   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_landmarks_project ON project_landmarks(project_id);
CREATE INDEX IF NOT EXISTS idx_landmarks_name    ON project_landmarks(landmark_name COLLATE NOCASE);

-- ─── project_amenities ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS project_amenities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    amenity     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amenities_project ON project_amenities(project_id);

-- ─── units ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS units (
    unit_id             TEXT PRIMARY KEY,
    project_id          TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,

    -- unit identity
    unit_type           TEXT,           -- "2 BHK", "Villa Serene Variant 1A"
    property_type       TEXT,           -- APARTMENT / VILLA / ROW_HOUSE / TENEMENT / PENTHOUSE
    bhk                 INTEGER,
    entrance_facing     TEXT,           -- North / South / East / West

    -- description (used for vector embedding)
    description         TEXT,

    -- areas
    carpet_area_sqft    REAL,
    super_builtup_sqft  REAL,
    balcony_area_sqft   REAL,
    wash_area_sqft      REAL,

    -- applicable buildings (comma-joined for apartments)
    applicable_buildings TEXT,

    -- computed room counts
    total_rooms         INTEGER DEFAULT 0,
    total_bedrooms      INTEGER DEFAULT 0,
    total_toilets       INTEGER DEFAULT 0,
    attached_bathrooms  INTEGER DEFAULT 0,

    -- key room areas (precomputed for fast filtering)
    master_bedroom_sqft REAL,
    drawing_room_sqft   REAL,
    kitchen_sqft        REAL,

    -- derived boolean feature flags (1/0)
    has_pooja_room      INTEGER DEFAULT 0,
    has_study_room      INTEGER DEFAULT 0,
    has_terrace         INTEGER DEFAULT 0,
    has_servant_room    INTEGER DEFAULT 0,
    has_garden          INTEGER DEFAULT 0,
    has_home_theatre    INTEGER DEFAULT 0,
    has_gym             INTEGER DEFAULT 0,
    has_dressing_room   INTEGER DEFAULT 0,
    has_store_room      INTEGER DEFAULT 0,
    has_courtyard       INTEGER DEFAULT 0,
    has_lobby           INTEGER DEFAULT 0,
    has_balcony         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_units_project      ON units(project_id);
CREATE INDEX IF NOT EXISTS idx_units_bhk          ON units(bhk);
CREATE INDEX IF NOT EXISTS idx_units_property_type ON units(property_type);

-- ─── Composite indexes for scale (20K+ projects) ─────────────────────────────
-- Most common query: filter by city → join to units by bhk + property_type
CREATE INDEX IF NOT EXISTS idx_projects_city        ON projects(city);
CREATE INDEX IF NOT EXISTS idx_projects_city_status ON projects(city, project_status);
CREATE INDEX IF NOT EXISTS idx_units_bhk_type       ON units(bhk, property_type);

-- ─── rooms ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rooms (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id             TEXT NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    project_id          TEXT NOT NULL,

    name                TEXT,
    room_type           TEXT,
    length              TEXT,
    width               TEXT,
    area_sqft           REAL,
    floor_level         TEXT,
    attached_bathroom   INTEGER,        -- 1/0/NULL for bedrooms
    has_balcony_access  INTEGER         -- 1/0/NULL for bedrooms/drawing rooms
);
CREATE INDEX IF NOT EXISTS idx_rooms_unit    ON rooms(unit_id);
CREATE INDEX IF NOT EXISTS idx_rooms_project ON rooms(project_id);
CREATE INDEX IF NOT EXISTS idx_rooms_type    ON rooms(room_type);

-- ─── FTS5 full-text search ────────────────────────────────────────────────────
-- Covers all text that a user might search: landmarks, amenities, descriptions,
-- city names, neighbourhood, developer names.
CREATE VIRTUAL TABLE IF NOT EXISTS projects_fts USING fts5(
    project_id    UNINDEXED,
    project_name,
    developer_name,
    city,
    neighbourhood,
    address,
    society_description,
    landmarks_text,
    amenities_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def create_all_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()
    logger.info("All SQLite tables and FTS5 index ready.")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


if __name__ == "__main__":
    conn = get_connection()
    create_all_tables(conn)
    conn.close()
    print(f"Schema created at: {settings.sqlite_path}")
