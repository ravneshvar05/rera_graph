"""
gemini.py  —  Real Estate Brochure Extraction Pipeline
=======================================================
Put PDFs in ./data/  →  run this  →  get JSONs in ./output/
Preprocessed PDFs saved to ./preprocessed/ for manual inspection.

Usage:
    python gemini.py                   # preprocess + extract all PDFs in ./data/
    python gemini.py --no-preprocess   # skip preprocessing, use raw PDFs

Dependencies:
    pip install google-genai pymupdf pillow python-dotenv pydantic
"""

import os
import re
import time
import uuid
import logging
import argparse
from enum import Enum
from typing import Optional, Literal, Union
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ── Import preprocessing ──────────────────────────────────────────────────────
try:
    from preprocess import preprocess_pdf
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
Path("pdf_tests/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_tests/logs/gemini_pdf_extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

MODEL = "gemini-2.5-flash"


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class RoomType(str, Enum):
    BEDROOM       = "BEDROOM"
    KITCHEN       = "KITCHEN"
    DRAWING_ROOM  = "DRAWING_ROOM"
    DINING        = "DINING"
    TOILET        = "TOILET"
    BATHROOM      = "BATHROOM"
    WC            = "WC"
    BALCONY       = "BALCONY"
    TERRACE       = "TERRACE"
    WASH_AREA     = "WASH_AREA"
    POOJA_ROOM    = "POOJA_ROOM"
    STORE_ROOM    = "STORE_ROOM"
    STUDY_ROOM    = "STUDY_ROOM"
    SERVANT_ROOM  = "SERVANT_ROOM"
    UTILITY_ROOM  = "UTILITY_ROOM"
    PASSAGE       = "PASSAGE"
    LOBBY         = "LOBBY"
    FOYER         = "FOYER"
    DRESSING_ROOM = "DRESSING_ROOM"
    COURTYARD     = "COURTYARD"
    OTHER         = "OTHER"

class ProjectStatus(str, Enum):
    UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"
    READY_TO_MOVE      = "READY_TO_MOVE"
    NEW_LAUNCH         = "NEW_LAUNCH"
    UNKNOWN            = "UNKNOWN"

class PropertyType(str, Enum):
    APARTMENT       = "APARTMENT"
    VILLA           = "VILLA"
    ROW_HOUSE       = "ROW_HOUSE"
    TENEMENT        = "TENEMENT"
    PENTHOUSE       = "PENTHOUSE"
    COMMERCIAL_SHOP = "COMMERCIAL_SHOP"
    OTHER           = "OTHER"

class BaseRoomSchema(BaseModel):
    name: str = Field(description="Exact room label on the plan e.g. 'Master Bedroom'")
    length: Optional[str] = Field(default=None, description="Raw length EXACTLY as printed e.g. \"10'-6\\\"\" or \"3.20\"")
    width:  Optional[str] = Field(default=None, description="Raw width EXACTLY as printed e.g. \"11'-0\\\"\" or \"3.45\"")
    area_sqft: Optional[float] = Field(default=None, description="Area in sqft — fill if printed, or will be computed from length x width")
    floor_level: Optional[str] = Field(default=None, description="Ground / First / Second / Terrace")

class PrimaryRoomSchema(BaseRoomSchema):
    room_type: Literal[RoomType.BEDROOM, RoomType.DRAWING_ROOM]
    attached_bathroom: Optional[bool] = Field(default=None, description="True if bathroom door opens directly into this room")
    has_balcony_access: Optional[bool] = Field(default=None, description="True if balcony is directly connected to this room")

class StandardRoomSchema(BaseRoomSchema):
    room_type: Literal[
        RoomType.KITCHEN, RoomType.DINING, RoomType.TOILET, RoomType.BATHROOM,
        RoomType.WC, RoomType.BALCONY, RoomType.TERRACE, RoomType.WASH_AREA,
        RoomType.POOJA_ROOM, RoomType.STORE_ROOM, RoomType.STUDY_ROOM,
        RoomType.SERVANT_ROOM, RoomType.UTILITY_ROOM, RoomType.PASSAGE,
        RoomType.LOBBY, RoomType.FOYER, RoomType.DRESSING_ROOM,
        RoomType.COURTYARD, RoomType.OTHER,
    ] = RoomType.OTHER

class SocietyLayoutSchema(BaseModel):
    description: Optional[str] = None
    total_apartment_blocks: Optional[int] = Field(default=None, description="ONLY for apartments")
    total_independent_villas_or_tenements: Optional[int] = Field(default=None, description="ONLY for villas/tenements/row-houses")
    building_names: list[str] = Field(default_factory=list, description="Tower names for apartments only. Empty for villas.")
    has_clubhouse: Optional[bool] = None
    has_park_or_garden: Optional[bool] = None
    has_swimming_pool: Optional[bool] = None
    has_sports_courts: Optional[bool] = None
    has_parking_area: Optional[bool] = None
    commercial_shops_included: Optional[bool] = None
    road_width_details: list[str] = Field(default_factory=list)

class FloorLayoutSchema(BaseModel):
    layout_name: Optional[str] = Field(default=None, description="ONLY for apartments e.g. 'Typical Floor 1st-10th'")
    total_units_on_floor: Optional[int] = None
    has_lifts: Optional[bool] = None
    has_staircases: Optional[bool] = None
    corridor_width: Optional[str] = None
    has_refuge_area: Optional[bool] = None

class UnitSchema(BaseModel):
    unit_type: str = Field(description="e.g. '2 BHK', '3 BHK + Maid', 'Villa Serene 1A'")
    property_type: PropertyType = Field(default=PropertyType.OTHER)
    applicable_buildings: list[str] = Field(default_factory=list, description="Empty for villas/row-houses")
    entrance_Facing: Optional[str] = Field(default=None, description="Compass direction main door faces e.g. 'East'. null if no compass found.")
    description: Optional[str] = Field(default=None, description="Short description of layout, vibe, or special features")
    bhk: Optional[int] = None
    carpet_area_sqft: Optional[float] = None
    balcony_area_sqft: Optional[float] = None
    wash_area_sqft: Optional[float] = None
    super_built_up_area_sqft: Optional[float] = None
    rooms: list[Union[PrimaryRoomSchema, StandardRoomSchema]] = Field(default_factory=list)

class LocationSchema(BaseModel):
    city: Optional[str] = None
    neighbourhood: Optional[str] = None
    address: Optional[str] = None
    pin_code: Optional[str] = None
    nearby_landmarks: list[str] = Field(default_factory=list, description="Real landmarks only: schools, hospitals, malls, transit. NOT competing residential societies.")

class BrochureData(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    brochure_file: str = Field(description="Filename of the processed brochure")
    project_name: Optional[str] = None
    developer_name: Optional[str] = None
    rera_registration_number: Optional[str] = None
    project_status: ProjectStatus = Field(default=ProjectStatus.UNKNOWN)
    possession_date: Optional[str] = None
    location: LocationSchema = Field(default_factory=LocationSchema)
    amenities: list[str] = Field(default_factory=list)
    society_layout: Optional[SocietyLayoutSchema] = None
    floor_layouts: list[FloorLayoutSchema] = Field(default_factory=list)
    units: list[UnitSchema] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT  (single-pass — proven to give complete output)
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = """
You are an expert real estate architect and data extraction AI.
Analyze this ENTIRE brochure PDF from first page to last page and extract ALL data.

CRITICAL INSTRUCTIONS:

1. UNITS — MOST IMPORTANT:
   - Extract EVERY unit/variant shown in the brochure. Do NOT leave units as an empty list.
   - Each variant (1A, 1B, 1C ...) is a completely SEPARATE unit with its OWN floor plans.
   - Read EACH floor plan image independently. Never copy room dimensions from one variant to another.

2. DIMENSIONS:
   - For every room, extract length and width as SEPARATE fields, EXACTLY as printed.
     Examples: "10'-6\"" or "3.05" — character for character, no conversion.
   - Dimensions are inside or beside each room boundary, often small or rotated — look carefully.
   - Sanity check: Master Bedroom > Bedroom > Kitchen ≈ Dining > Toilet (toilet width ≤ 5').
   - If you genuinely cannot read a dimension, use null. A null is better than a wrong number.

3. ROOM AREA:
   - If area_sqft is explicitly printed for a room, fill it.
   - If not printed but length and width are available, compute: area_sqft = length_ft × width_ft.
     Convert metric to feet first if needed (1 metre = 3.281 ft).

4. MISSING DATA:
   - Use JSON null (never the string "null") for anything not found.
   - super_built_up_area_sqft and carpet_area_sqft come ONLY from printed schedule tables.

5. ROOM CONNECTIONS (read the floor plan visually):
   - attached_bathroom = true only if a bathroom door opens directly into that bedroom/drawing room.
   - has_balcony_access = true only if a balcony is directly connected to that bedroom/drawing room.

6. ENTRANCE FACING:
   - Find compass symbols and trace which direction the main entrance door faces.
   - For apartments: door facing the corridor. For villas: door facing the street/plot.
   - Set null ONLY if the entire brochure has no compass symbol at all.

7. VILLA vs APARTMENT:
   - Floor layouts (lifts, corridors) → apartments only.
   - Villas/Row-houses → applicable_buildings must be [].

8. NEARBY LANDMARKS:
   - Only real landmarks: schools, hospitals, malls, transit hubs.
   - Do NOT include other residential society names from the location map.

9. AMENITIES:
   - List every amenity mentioned anywhere in the brochure.

Output strictly as a JSON object matching the schema. Cover every page, every unit, every room.
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_ft(raw: str) -> Optional[float]:
    """
    Try to parse a raw dimension string into feet (float).
    Handles formats like: 10'-6", 10'6", 10-6, 3.05 (metres), 10.5
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.strip()

    # Format: 10'-6" or 10'6" or 10'0"
    m = re.match(r"(\d+)['\u2019][\s\-]?(\d+)[\"\u201d]?", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12

    # Format: 10-6 (feet-inches with dash)
    m = re.match(r"^(\d+)-(\d+)$", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12

    # Plain decimal — could be metres or feet
    m = re.match(r"^(\d+\.?\d*)$", raw)
    if m:
        val = float(m.group(1))
        # Heuristic: if < 15 assume metres, convert to feet
        return val * 3.281 if val < 15 else val

    return None


def compute_room_areas(data: dict) -> dict:
    """
    For any room that has length + width but no area_sqft, compute and fill it.
    """
    for unit in data.get("units", []):
        for room in unit.get("rooms", []):
            if room.get("area_sqft") is not None:
                continue  # already filled
            l = parse_ft(room.get("length") or "")
            w = parse_ft(room.get("width") or "")
            if l and w:
                room["area_sqft"] = round(l * w, 1)
    return data


def postprocess(data: dict) -> dict:
    """Sanity checks + compute missing room areas."""
    data = compute_room_areas(data)

    units = data.get("units", [])
    fingerprints: dict[str, list[str]] = {}

    for unit in units:
        bhk   = unit.get("bhk") or 0
        sba   = unit.get("super_built_up_area_sqft")
        ptype = unit.get("property_type", "")
        utype = unit.get("unit_type", "?")

        # Flag impossible SBA
        if sba and bhk >= 4 and ptype in ("VILLA", "ROW_HOUSE", "TENEMENT") and sba < 1500:
            logging.warning(f"  [SANITY] SBA={sba} sqft too small for {bhk}BHK {ptype} '{utype}'. Nulling.")
            unit["super_built_up_area_sqft"] = None

        # Clear zero areas
        for room in unit.get("rooms", []):
            if room.get("area_sqft") == 0:
                room["area_sqft"] = None

        # Detect dimension copying across variants
        sig = "|".join(
            f"{r.get('name')}:{r.get('length')}x{r.get('width')}"
            for r in unit.get("rooms", []) if r.get("length")
        )
        fingerprints.setdefault(sig, []).append(utype)

    for sig, utypes in fingerprints.items():
        if len(utypes) > 1 and sig:
            logging.warning(f"  [SANITY] Identical room dims across variants {utypes} — possible hallucination.")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXTRACTION  (same proven structure as original, with fixes)
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(client: genai.Client, pdf_path: Path) -> str | None:
    logging.info(f"  Uploading: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")

    try:
        pdf_file = client.files.upload(file=str(pdf_path), config={"mime_type": "application/pdf"})

        logging.info("  Waiting for Gemini to process file ...")
        while pdf_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            pdf_file = client.files.get(name=pdf_file.name)
        print()

        if pdf_file.state.name == "FAILED":
            logging.error("  File processing failed on the server.")
            return None

        logging.info(f"  Calling {MODEL} ...")
        response = client.models.generate_content(
            model=MODEL,
            contents=[pdf_file, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BrochureData,
                temperature=0.0,
                max_output_tokens=32000,   # high limit — prevents cut-off mid-JSON
            ),
        )

        try:
            client.files.delete(name=pdf_file.name)
        except Exception:
            pass

        return response.text

    except Exception as e:
        logging.error(f"  API call failed: {str(e)[:400]}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Extract JSON from real estate brochures using Gemini.")
    p.add_argument("--no-preprocess", action="store_true", help="Skip PDF preprocessing")
    p.add_argument("--dpi",      type=int,  default=250,            help="Preprocessing DPI (default: 250)")
    p.add_argument("--quality",  type=int,  default=82,             help="JPEG quality (default: 82)")
    p.add_argument("--overwrite-preprocessed", action="store_true", help="Re-preprocess existing files")
    p.add_argument("--input-dir",  type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--prep-dir",   type=Path, default=Path("preprocessed"))
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prep_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_files:
        logging.error(f"No PDFs found in '{args.input_dir}/'.")
        return

    use_preprocess = (not args.no_preprocess) and PREPROCESS_AVAILABLE
    logging.info(f"Found {len(pdf_files)} PDF(s) | Preprocessing: {'ON' if use_preprocess else 'OFF'}")

    import json
    success = 0

    for idx, pdf_path in enumerate(pdf_files):
        logging.info(f"\n--- [{idx+1}/{len(pdf_files)}] {pdf_path.name} ---")

        # Step 1: Preprocess
        if use_preprocess:
            logging.info("  Step 1: Preprocessing ...")
            try:
                work_path = preprocess_pdf(
                    pdf_path=pdf_path,
                    output_dir=args.prep_dir,
                    dpi=args.dpi,
                    quality=args.quality,
                    enhance=True,
                    overwrite=args.overwrite_preprocessed,
                )
                logging.info(f"  Preprocessed PDF -> {work_path}  (open to verify quality)")
            except Exception as e:
                logging.error(f"  Preprocessing failed: {e} — using raw PDF.")
                work_path = pdf_path
        else:
            work_path = pdf_path

        # Step 2: Extract
        logging.info("  Step 2: Extracting ...")
        result_text = process_pdf(client, work_path)

        if not result_text:
            logging.error(f"  SKIPPED — no result for {pdf_path.name}")
            continue

        # Step 3: Postprocess + save
        try:
            data = json.loads(result_text)
            data["brochure_file"] = pdf_path.name
            data = postprocess(data)
            final_json = json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            logging.error(f"  JSON parse error: {e} — saving raw output.")
            final_json = result_text

        out_file = args.output_dir / f"{pdf_path.stem}.json"
        out_file.write_text(final_json, encoding="utf-8")
        logging.info(f"  SAVED -> {out_file}")
        success += 1

    logging.info(f"\nDone! {success}/{len(pdf_files)} extracted successfully.")


if __name__ == "__main__":
    main()