# """
# gemini.py  —  Real Estate Brochure Extraction Pipeline
# =======================================================
# Put PDFs in ./data/  ->  run this  ->  get JSONs in ./output/
# Preprocessed PDFs saved to ./preprocessed/ for manual inspection.

# Usage:
#     python gemini.py                   # preprocess + extract all PDFs in ./data/
#     python gemini.py --no-preprocess   # skip preprocessing, use raw PDFs

# Dependencies:
#     pip install google-genai pymupdf pillow python-dotenv pydantic
# """

# import os
# import re
# import time
# import uuid
# import json
# import logging
# import argparse
# from enum import Enum
# from typing import Optional, Literal, Union
# from pathlib import Path
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# from pydantic import BaseModel, Field

# try:
#     from preprocess import preprocess_pdf
#     PREPROCESS_AVAILABLE = True
# except ImportError:
#     PREPROCESS_AVAILABLE = False

# load_dotenv()

# Path("pdf_tests/logs").mkdir(parents=True, exist_ok=True)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("pdf_tests/logs/gemini_pdf_extraction.log", encoding="utf-8"),
#         logging.StreamHandler(),
#     ],
# )

# MODEL = "gemini-2.5-flash"


# # ══════════════════════════════════════════════════════════════════════════════
# # SCHEMA  (project_id excluded — we inject UUID ourselves after extraction)
# # ══════════════════════════════════════════════════════════════════════════════

# class RoomType(str, Enum):
#     BEDROOM = "BEDROOM"; KITCHEN = "KITCHEN"; DRAWING_ROOM = "DRAWING_ROOM"
#     DINING = "DINING"; TOILET = "TOILET"; BATHROOM = "BATHROOM"; WC = "WC"
#     BALCONY = "BALCONY"; TERRACE = "TERRACE"; WASH_AREA = "WASH_AREA"
#     POOJA_ROOM = "POOJA_ROOM"; STORE_ROOM = "STORE_ROOM"; STUDY_ROOM = "STUDY_ROOM"
#     SERVANT_ROOM = "SERVANT_ROOM"; UTILITY_ROOM = "UTILITY_ROOM"; PASSAGE = "PASSAGE"
#     LOBBY = "LOBBY"; FOYER = "FOYER"; DRESSING_ROOM = "DRESSING_ROOM"
#     COURTYARD = "COURTYARD"; OTHER = "OTHER"

# class ProjectStatus(str, Enum):
#     UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"; READY_TO_MOVE = "READY_TO_MOVE"
#     NEW_LAUNCH = "NEW_LAUNCH"; UNKNOWN = "UNKNOWN"

# class PropertyType(str, Enum):
#     APARTMENT = "APARTMENT"; VILLA = "VILLA"; ROW_HOUSE = "ROW_HOUSE"
#     TENEMENT = "TENEMENT"; PENTHOUSE = "PENTHOUSE"
#     COMMERCIAL_SHOP = "COMMERCIAL_SHOP"; OTHER = "OTHER"

# class BaseRoomSchema(BaseModel):
#     name: str
#     length: Optional[str] = None
#     width: Optional[str] = None
#     area_sqft: Optional[float] = None
#     floor_level: Optional[str] = None

# class PrimaryRoomSchema(BaseRoomSchema):
#     room_type: Literal[RoomType.BEDROOM, RoomType.DRAWING_ROOM]
#     attached_bathroom: Optional[bool] = None
#     has_balcony_access: Optional[bool] = None

# class StandardRoomSchema(BaseRoomSchema):
#     room_type: Literal[
#         RoomType.KITCHEN, RoomType.DINING, RoomType.TOILET, RoomType.BATHROOM,
#         RoomType.WC, RoomType.BALCONY, RoomType.TERRACE, RoomType.WASH_AREA,
#         RoomType.POOJA_ROOM, RoomType.STORE_ROOM, RoomType.STUDY_ROOM,
#         RoomType.SERVANT_ROOM, RoomType.UTILITY_ROOM, RoomType.PASSAGE,
#         RoomType.LOBBY, RoomType.FOYER, RoomType.DRESSING_ROOM,
#         RoomType.COURTYARD, RoomType.OTHER,
#     ] = RoomType.OTHER

# class SocietyLayoutSchema(BaseModel):
#     description: Optional[str] = None
#     total_apartment_blocks: Optional[int] = None
#     total_independent_villas_or_tenements: Optional[int] = None
#     building_names: list[str] = Field(default_factory=list)
#     has_clubhouse: Optional[bool] = None
#     has_park_or_garden: Optional[bool] = None
#     has_swimming_pool: Optional[bool] = None
#     has_sports_courts: Optional[bool] = None
#     has_parking_area: Optional[bool] = None
#     commercial_shops_included: Optional[bool] = None
#     road_width_details: list[str] = Field(default_factory=list)

# class FloorLayoutSchema(BaseModel):
#     layout_name: Optional[str] = None
#     total_units_on_floor: Optional[int] = None
#     has_lifts: Optional[bool] = None
#     has_staircases: Optional[bool] = None
#     corridor_width: Optional[str] = None
#     has_refuge_area: Optional[bool] = None

# class UnitSchema(BaseModel):
#     unit_type: str
#     property_type: PropertyType = PropertyType.OTHER
#     applicable_buildings: list[str] = Field(default_factory=list)
#     entrance_Facing: Optional[str] = None
#     description: Optional[str] = None
#     bhk: Optional[int] = None
#     carpet_area_sqft: Optional[float] = None
#     balcony_area_sqft: Optional[float] = None
#     wash_area_sqft: Optional[float] = None
#     super_built_up_area_sqft: Optional[float] = None
#     rooms: list[Union[PrimaryRoomSchema, StandardRoomSchema]] = Field(default_factory=list)

# class LocationSchema(BaseModel):
#     city: Optional[str] = None
#     neighbourhood: Optional[str] = None
#     address: Optional[str] = None
#     pin_code: Optional[str] = None
#     nearby_landmarks: list[str] = Field(default_factory=list)


# # ══════════════════════════════════════════════════════════════════════════════
# # PROMPT
# # ══════════════════════════════════════════════════════════════════════════════

# PROMPT = """
# You are an expert real estate architect and data extraction AI.
# Analyze this ENTIRE brochure PDF from first page to last page and extract ALL data.

# OUTPUT FORMAT: Return a single complete JSON object. It must be valid JSON from
# opening {{ to closing }}. Do not stop early. Do not truncate. Every unit, every
# room, every field must be included before you close the JSON.

# CRITICAL INSTRUCTIONS:

# 1. UNITS:
#    - Extract EVERY unit/variant. Do NOT leave units as an empty list.
#    - Each variant (1A, 1B, 1C ...) is SEPARATE with its OWN floor plans and dimensions.
#    - Never copy dimensions from one variant to another — read each floor plan fresh.

# 2. DIMENSIONS:
#    - Extract length and width as SEPARATE fields, EXACTLY as printed (e.g. "10'-6\\"").
#    - Look carefully — dimensions are often small or rotated inside room boundaries.
#    - Sanity: Master Bedroom > Bedroom > Kitchen ~ Dining > Toilet (width <= 5').
#    - Cannot read a dimension -> use null. Null is better than a wrong number.

# 3. ROOM AREA:
#    - Fill area_sqft if explicitly printed for that room.
#    - If not printed but length and width are present, compute: length_ft x width_ft.
#    - Convert metric to feet first if needed (1 metre = 3.281 ft).

# 4. MISSING DATA:
#    - Use JSON null (not the string "null", not "NA", not "None") for anything missing.
#    - super_built_up_area_sqft and carpet_area_sqft: ONLY from printed schedule tables.

# 5. ROOM CONNECTIONS:
#    - attached_bathroom = true only if bathroom door opens directly into that bedroom.
#    - has_balcony_access = true only if balcony is directly connected to that room.

# 6. ENTRANCE FACING:
#    - Trace compass symbols to find which direction the main entrance faces.
#    - null ONLY if the entire brochure has no compass symbol.

# 7. VILLA vs APARTMENT:
#    - Floor layouts (lifts, corridors) for apartments only.
#    - Villas/Row-houses: applicable_buildings must be [].

# 8. NEARBY LANDMARKS:
#    - Only real landmarks: schools, hospitals, malls, transit hubs.
#    - Exclude residential society names from the location map.
#    - Correct obvious OCR errors in names (e.g. "ISION TEMPLE" -> "ISKCON TEMPLE").

# 9. AMENITIES: List every amenity mentioned in the brochure.

# JSON structure required:
# {
#   "brochure_file": "...",
#   "project_name": "...",
#   "developer_name": "...",
#   "rera_registration_number": null,
#   "project_status": "UNKNOWN",
#   "possession_date": null,
#   "location": { "city": "...", "neighbourhood": "...", "address": null, "pin_code": null, "nearby_landmarks": [] },
#   "amenities": [],
#   "society_layout": { ... },
#   "floor_layouts": [],
#   "units": [
#     {
#       "unit_type": "...",
#       "property_type": "VILLA",
#       "applicable_buildings": [],
#       "entrance_Facing": "East",
#       "description": "...",
#       "bhk": 4,
#       "carpet_area_sqft": null,
#       "super_built_up_area_sqft": null,
#       "rooms": [
#         { "name": "Master Bedroom", "room_type": "BEDROOM", "length": "12'-0\\"", "width": "14'-0\\"", "area_sqft": 168.0, "floor_level": "First", "attached_bathroom": true, "has_balcony_access": true }
#       ]
#     }
#   ]
# }
# """


# # ══════════════════════════════════════════════════════════════════════════════
# # JSON REPAIR  — fixes truncated output
# # ══════════════════════════════════════════════════════════════════════════════

# def repair_truncated_json(text: str) -> str:
#     """
#     If Gemini cuts off mid-JSON, close all open brackets/braces so the
#     output is at least parseable (last unit/room may be incomplete but
#     everything before it is saved).
#     """
#     text = text.strip()
    
#     # Strip markdown block if model wrapped the JSON (common without response_schema)
#     if text.startswith("```json"):
#         text = text[7:].strip()
#     if text.endswith("```"):
#         text = text[:-3].strip()

#     # Remove any trailing comma before we close
#     text = re.sub(r",\s*$", "", text)

#     # Count unclosed brackets
#     stack = []
#     in_string = False
#     escape = False
#     for ch in text:
#         if escape:
#             escape = False
#             continue
#         if ch == "\\" and in_string:
#             escape = True
#             continue
#         if ch == '"':
#             in_string = not in_string
#             continue
#         if in_string:
#             continue
#         if ch in "{[":
#             stack.append(ch)
#         elif ch in "}]":
#             if stack:
#                 stack.pop()

#     # If it cut off in the middle of a string value, close the quote first!
#     if in_string:
#         text += '"'

#     # Close in reverse order
#     closing = {"[": "]", "{": "}"}
#     for opener in reversed(stack):
#         text += closing[opener]

#     return text


# # ══════════════════════════════════════════════════════════════════════════════
# # POST-PROCESSING
# # ══════════════════════════════════════════════════════════════════════════════

# _NULL_STRINGS = {"null", "na", "none", "n/a", "nil", "undefined", "-"}

# def clean_null_strings(obj):
#     """Recursively replace null-like strings with None."""
#     if isinstance(obj, dict):
#         return {k: clean_null_strings(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [clean_null_strings(i) for i in obj]
#     if isinstance(obj, str) and obj.strip().lower() in _NULL_STRINGS:
#         return None
#     return obj


# def parse_ft(raw: str) -> Optional[float]:
#     if not raw:
#         return None
#     raw = raw.strip()
#     m = re.match(r"(\d+)['\u2019][\s\-]?(\d+)[\"\u201d]?", raw)
#     if m:
#         return int(m.group(1)) + int(m.group(2)) / 12
#     m = re.match(r"^(\d+)-(\d+)$", raw)
#     if m:
#         return int(m.group(1)) + int(m.group(2)) / 12
#     m = re.match(r"^(\d+\.?\d*)$", raw)
#     if m:
#         val = float(m.group(1))
#         return val * 3.281 if val < 15 else val
#     return None


# def compute_room_areas(data: dict) -> dict:
#     for unit in data.get("units", []):
#         for room in unit.get("rooms", []):
#             if room.get("area_sqft") is not None:
#                 continue
#             l = parse_ft(room.get("length") or "")
#             w = parse_ft(room.get("width") or "")
#             if l and w:
#                 room["area_sqft"] = round(l * w, 1)
#     return data


# def fix_hallucinated_dimensions(data: dict) -> dict:
#     units = data.get("units", [])

#     def sig(unit):
#         return "|".join(
#             f"{r.get('name')}:{r.get('length')}x{r.get('width')}"
#             for r in unit.get("rooms", [])
#             if r.get("length") or r.get("width")
#         )

#     seen: dict[str, str] = {}
#     for unit in units:
#         s = sig(unit)
#         if not s:
#             continue
#         utype = unit.get("unit_type", "?")
#         if s in seen:
#             logging.warning(
#                 f"  [HALLUCINATION] '{utype}' has same dims as '{seen[s]}' — clearing."
#             )
#             for room in unit.get("rooms", []):
#                 room["length"] = None
#                 room["width"] = None
#                 room["area_sqft"] = None
#         else:
#             seen[s] = utype
#     return data


# def postprocess(data: dict, filename: str) -> dict:
#     data = clean_null_strings(data)
#     data["project_id"] = str(uuid.uuid4())
#     data["brochure_file"] = filename
#     data = fix_hallucinated_dimensions(data)
#     data = compute_room_areas(data)

#     for unit in data.get("units", []):
#         bhk = unit.get("bhk") or 0
#         sba = unit.get("super_built_up_area_sqft")
#         ptype = unit.get("property_type", "")
#         utype = unit.get("unit_type", "?")
#         if sba and bhk >= 4 and ptype in ("VILLA", "ROW_HOUSE", "TENEMENT") and sba < 1500:
#             logging.warning(f"  [SANITY] SBA={sba} too small for {bhk}BHK {ptype} '{utype}'. Nulling.")
#             unit["super_built_up_area_sqft"] = None
#         for room in unit.get("rooms", []):
#             if room.get("area_sqft") == 0:
#                 room["area_sqft"] = None

#     return data


# # ══════════════════════════════════════════════════════════════════════════════
# # EXTRACTION
# # ══════════════════════════════════════════════════════════════════════════════

# def process_pdf(client: genai.Client, pdf_path: Path) -> str | None:
#     logging.info(f"  Uploading: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")

#     try:
#         pdf_file = client.files.upload(file=str(pdf_path), config={"mime_type": "application/pdf"})

#         logging.info("  Waiting for Gemini to process file ...")
#         while pdf_file.state.name == "PROCESSING":
#             print(".", end="", flush=True)
#             time.sleep(2)
#             pdf_file = client.files.get(name=pdf_file.name)
#         print()

#         if pdf_file.state.name == "FAILED":
#             logging.error("  File processing failed on the server.")
#             return None

#         logging.info(f"  Calling {MODEL} ...")
#         response = client.models.generate_content(
#             model=MODEL,
#             contents=[pdf_file, PROMPT],
#             config=types.GenerateContentConfig(
#                 # No response_schema — schema enforcement has an internal token cap
#                 # that cuts off large JSONs. We validate the structure ourselves.
#                 response_mime_type="application/json",
#                 temperature=0.0,
#                 max_output_tokens=65000,
#             ),
#         )

#         try:
#             client.files.delete(name=pdf_file.name)
#         except Exception:
#             pass

#         return response.text

#     except Exception as e:
#         logging.error(f"  API call failed: {str(e)[:400]}")
#         return None


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(description="Extract JSON from real estate brochures using Gemini.")
#     p.add_argument("--no-preprocess",          action="store_true", help="Skip PDF preprocessing")
#     p.add_argument("--dpi",     type=int, default=250,             help="Preprocessing DPI (default: 250)")
#     p.add_argument("--quality", type=int, default=82,              help="JPEG quality (default: 82)")
#     p.add_argument("--overwrite-preprocessed", action="store_true", help="Re-preprocess existing files")
#     p.add_argument("--input-dir",  type=Path, default=Path("data"))
#     p.add_argument("--output-dir", type=Path, default=Path("output"))
#     p.add_argument("--prep-dir",   type=Path, default=Path("preprocessed"))
#     return p.parse_args()


# def main():
#     args = parse_args()

#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         logging.error("GEMINI_API_KEY not found in .env file.")
#         return

#     client = genai.Client(api_key=api_key)
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     args.prep_dir.mkdir(parents=True, exist_ok=True)

#     pdf_files = sorted(args.input_dir.glob("*.pdf"))
#     if not pdf_files:
#         logging.error(f"No PDFs found in '{args.input_dir}/'.")
#         return

#     use_preprocess = (not args.no_preprocess) and PREPROCESS_AVAILABLE
#     logging.info(f"Found {len(pdf_files)} PDF(s) | Preprocessing: {'ON' if use_preprocess else 'OFF'}")

#     success = 0

#     for idx, pdf_path in enumerate(pdf_files):
#         logging.info(f"\n--- [{idx+1}/{len(pdf_files)}] {pdf_path.name} ---")

#         # Step 1: Preprocess
#         if use_preprocess:
#             logging.info("  Step 1: Preprocessing ...")
#             try:
#                 work_path = preprocess_pdf(
#                     pdf_path=pdf_path,
#                     output_dir=args.prep_dir,
#                     dpi=args.dpi,
#                     quality=args.quality,
#                     enhance=True,
#                     overwrite=args.overwrite_preprocessed,
#                 )
#                 logging.info(f"  Preprocessed PDF -> {work_path}")
#             except Exception as e:
#                 logging.error(f"  Preprocessing failed: {e} — using raw PDF.")
#                 work_path = pdf_path
#         else:
#             work_path = pdf_path

#         # Step 2: Extract
#         logging.info("  Step 2: Extracting ...")
#         result_text = process_pdf(client, work_path)

#         if not result_text:
#             logging.error(f"  SKIPPED — no result for {pdf_path.name}")
#             continue

#         # Step 3: Parse — auto-repair if truncated
#         try:
#             data = json.loads(result_text)
#         except json.JSONDecodeError:
#             logging.warning("  JSON truncated — attempting auto-repair ...")
#             repaired = repair_truncated_json(result_text)
#             try:
#                 data = json.loads(repaired)
#                 logging.info("  Auto-repair succeeded.")
#             except json.JSONDecodeError as e:
#                 logging.error(f"  Repair failed: {e} — saving raw output.")
#                 (args.output_dir / f"{pdf_path.stem}_raw.json").write_text(result_text, encoding="utf-8")
#                 continue

#         # Step 4: Postprocess + save
#         data = postprocess(data, pdf_path.name)
#         out_file = args.output_dir / f"{pdf_path.stem}.json"
#         out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
#         logging.info(f"  SAVED -> {out_file}")
#         success += 1

#     logging.info(f"\nDone! {success}/{len(pdf_files)} extracted successfully.")


# if __name__ == "__main__":
#     main()




"""
gemini.py  —  Real Estate Brochure Extraction Pipeline
=======================================================
Put PDFs in ./data/  ->  run this  ->  get JSONs in ./output/

Strategy (solves truncation while keeping Pydantic schema):
  Pass 1 — full PDF, no schema: extract project-level data + list of unit names.
            Small output, never truncates.
  Pass 2 — full PDF + Pydantic schema, one unit at a time: extract each unit
            individually. Each call is small so schema enforcement never hits
            the token cap.
  Merge  — combine all unit results into one final JSON.

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
import json
import logging
import argparse
from enum import Enum
from typing import Optional, Literal, Union
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from preprocess import preprocess_pdf
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

load_dotenv()

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
    BEDROOM = "BEDROOM"; KITCHEN = "KITCHEN"; DRAWING_ROOM = "DRAWING_ROOM"
    DINING = "DINING"; TOILET = "TOILET"; BATHROOM = "BATHROOM"; WC = "WC"
    BALCONY = "BALCONY"; TERRACE = "TERRACE"; WASH_AREA = "WASH_AREA"
    POOJA_ROOM = "POOJA_ROOM"; STORE_ROOM = "STORE_ROOM"; STUDY_ROOM = "STUDY_ROOM"
    SERVANT_ROOM = "SERVANT_ROOM"; UTILITY_ROOM = "UTILITY_ROOM"; PASSAGE = "PASSAGE"
    LOBBY = "LOBBY"; FOYER = "FOYER"; DRESSING_ROOM = "DRESSING_ROOM"
    COURTYARD = "COURTYARD"; OTHER = "OTHER"

class ProjectStatus(str, Enum):
    UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"; READY_TO_MOVE = "READY_TO_MOVE"
    NEW_LAUNCH = "NEW_LAUNCH"; UNKNOWN = "UNKNOWN"

class PropertyType(str, Enum):
    APARTMENT = "APARTMENT"; VILLA = "VILLA"; ROW_HOUSE = "ROW_HOUSE"
    TENEMENT = "TENEMENT"; PENTHOUSE = "PENTHOUSE"
    COMMERCIAL_SHOP = "COMMERCIAL_SHOP"; OTHER = "OTHER"

class BaseRoomSchema(BaseModel):
    name: str = Field(description="Exact room label e.g. 'Master Bedroom'")
    length: Optional[str] = Field(default=None, description="Raw length EXACTLY as printed e.g. \"10'-6\\\"\"")
    width: Optional[str] = Field(default=None, description="Raw width EXACTLY as printed e.g. \"11'-0\\\"\"")
    area_sqft: Optional[float] = Field(default=None, description="Fill if printed; computed later if null")
    floor_level: Optional[str] = Field(default=None, description="Ground / First / Second / Terrace")

class PrimaryRoomSchema(BaseRoomSchema):
    room_type: Literal[RoomType.BEDROOM, RoomType.DRAWING_ROOM]
    attached_bathroom: Optional[bool] = Field(default=None)
    has_balcony_access: Optional[bool] = Field(default=None)

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
    total_apartment_blocks: Optional[int] = None
    total_independent_villas_or_tenements: Optional[int] = None
    building_names: list[str] = Field(default_factory=list)
    has_clubhouse: Optional[bool] = None
    has_park_or_garden: Optional[bool] = None
    has_swimming_pool: Optional[bool] = None
    has_sports_courts: Optional[bool] = None
    has_parking_area: Optional[bool] = None
    commercial_shops_included: Optional[bool] = None
    road_width_details: list[str] = Field(default_factory=list)

class FloorLayoutSchema(BaseModel):
    layout_name: Optional[str] = None
    total_units_on_floor: Optional[int] = None
    has_lifts: Optional[bool] = None
    has_staircases: Optional[bool] = None
    corridor_width: Optional[str] = None
    has_refuge_area: Optional[bool] = None

# Schema for a single unit — used in Pass 2
class UnitSchema(BaseModel):
    unit_type: str
    property_type: PropertyType = PropertyType.OTHER
    applicable_buildings: list[str] = Field(default_factory=list)
    entrance_Facing: Optional[str] = None
    description: Optional[str] = None
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
    nearby_landmarks: list[str] = Field(default_factory=list)

# Schema for project-level data only (no units) — used in Pass 1
class ProjectDataSchema(BaseModel):
    brochure_file: str
    unit_names: list[str] = Field(
        default_factory=list,
        description="List of every unit/variant name exactly as labelled in the brochure e.g. ['Villa Serene 1A', 'Villa Serene 1B', 'Villa Exotica']"
    )
    project_name: Optional[str] = None
    developer_name: Optional[str] = None
    rera_registration_number: Optional[str] = None
    project_status: ProjectStatus = ProjectStatus.UNKNOWN
    possession_date: Optional[str] = None
    location: LocationSchema = Field(default_factory=LocationSchema)
    amenities: list[str] = Field(default_factory=list)
    society_layout: Optional[SocietyLayoutSchema] = None
    floor_layouts: list[FloorLayoutSchema] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PASS1_PROMPT = """
You are a real estate data extraction AI.
Analyze this entire brochure PDF from the very first page to the very last page and extract:

1. All project-level information: name, developer, location, amenities, society layout, RERA number, status.
2. A complete list of EVERY unit/variant name exactly as labelled (e.g. "Villa Serene 1A", "3 BHK Type A").
   CRITICAL: You must scan EVERY single page of the brochure for floor plans. Do NOT stop after finding 3 or 4 variants. There are often 6+ variants listed on later pages. List them ALL.

For nearby_landmarks: only real landmarks (schools, hospitals, malls, transit).
Exclude residential society names. Correct obvious OCR errors in names.

For unit_names: just the names as a list — no room details needed here.
"""

def make_pass2_prompt(unit_name: str) -> str:
    return f"""
You are a real estate data extraction AI.
Extract complete data for this ONE specific unit from the brochure: "{unit_name}"

Find the floor plan(s) for "{unit_name}" and extract every room on every floor.

RULES:
1. DIMENSIONS: Extract length and width as SEPARATE fields, EXACTLY as printed.
   Look carefully — dimensions are often small or rotated inside room boundaries.
   Sanity: Master Bedroom > Bedroom > Kitchen ~ Dining > Toilet (width <= 5').
   Cannot read -> null. Null is better than a wrong number.

2. ROOM AREA: Fill area_sqft if explicitly printed. If not, leave null (computed later).

3. ROOM CONNECTIONS:
   attached_bathroom = true only if bathroom door opens directly into that bedroom.
   has_balcony_access = true only if balcony is directly connected to that room.

4. ENTRANCE FACING: Trace compass symbols to find which direction main entrance faces.
   null only if no compass symbol exists anywhere in the brochure.

5. MISSING DATA: JSON null (not the string "null", not "NA", not "None").

6. applicable_buildings: empty [] for villas/row-houses.

7. super_built_up_area_sqft / carpet_area_sqft: ONLY from printed schedule tables.
"""


# ══════════════════════════════════════════════════════════════════════════════
# JSON REPAIR / POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def repair_truncated_json(text: str) -> str:
    """
    If Gemini cuts off mid-JSON, close all open brackets/braces so the
    output is at least parseable.
    """
    text = text.strip()
    
    # Strip markdown block if model wrapped the JSON
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Remove any trailing comma before we close
    text = re.sub(r",\s*$", "", text)

    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    if in_string:
        text += '"'

    closing = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        text += closing[opener]

    return text

_NULL_STRINGS = {"null", "na", "none", "n/a", "nil", "undefined", "-"}


def clean_null_strings(obj):
    if isinstance(obj, dict):
        return {k: clean_null_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_null_strings(i) for i in obj]
    if isinstance(obj, str) and obj.strip().lower() in _NULL_STRINGS:
        return None
    return obj


def parse_ft(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = raw.strip()
    m = re.match(r"(\d+)['\u2019][\s\-]?(\d+)[\"\u201d]?", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12
    m = re.match(r"^(\d+)-(\d+)$", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12
    m = re.match(r"^(\d+\.?\d*)$", raw)
    if m:
        val = float(m.group(1))
        return val * 3.281 if val < 15 else val
    return None


def compute_room_areas(units: list) -> list:
    for unit in units:
        for room in unit.get("rooms", []):
            if room.get("area_sqft") is not None:
                continue
            l = parse_ft(room.get("length") or "")
            w = parse_ft(room.get("width") or "")
            if l and w:
                room["area_sqft"] = round(l * w, 1)
    return units


def fix_hallucinated_dimensions(units: list) -> list:
    def sig(unit):
        return "|".join(
            f"{r.get('name')}:{r.get('length')}x{r.get('width')}"
            for r in unit.get("rooms", [])
            if r.get("length") or r.get("width")
        )
    seen: dict[str, str] = {}
    for unit in units:
        s = sig(unit)
        if not s:
            continue
        utype = unit.get("unit_type", "?")
        if s in seen:
            logging.warning(f"  [HALLUCINATION] '{utype}' same dims as '{seen[s]}' — clearing.")
            for room in unit.get("rooms", []):
                room["length"] = None
                room["width"] = None
                room["area_sqft"] = None
        else:
            seen[s] = utype
    return units


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CALLS
# ══════════════════════════════════════════════════════════════════════════════

def upload_and_wait(client: genai.Client, pdf_path: Path):
    logging.info(f"  Uploading: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    pdf_file = client.files.upload(file=str(pdf_path), config={"mime_type": "application/pdf"})
    while pdf_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        pdf_file = client.files.get(name=pdf_file.name)
    print()
    if pdf_file.state.name == "FAILED":
        raise RuntimeError("Gemini file processing failed.")
    return pdf_file


def call_gemini(client, contents, schema=None, max_tokens=8192) -> Optional[str]:
    """Single Gemini API call. Returns response text or None on failure."""
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=max_tokens,
        )
        if schema:
            config.response_schema = schema
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        return response.text
    except Exception as e:
        logging.error(f"  Gemini call failed: {str(e)[:300]}")
        return None


def process_pdf(client: genai.Client, pdf_path: Path) -> Optional[dict]:
    """
    Full two-pass extraction for one PDF.
    Returns merged dict or None on failure.
    """
    # Upload once — reuse the same file reference for all calls
    try:
        pdf_file = upload_and_wait(client, pdf_path)
    except Exception as e:
        logging.error(f"  Upload failed: {e}")
        return None

    # ── Pass 1: project data + unit name list ─────────────────────────────
    logging.info("  Pass 1: extracting project data + unit names ...")
    raw1 = call_gemini(
        client,
        contents=[pdf_file, PASS1_PROMPT],
        schema=ProjectDataSchema,
        max_tokens=4096,   # project data is always small
    )
    if not raw1:
        logging.error("  Pass 1 failed.")
        try: client.files.delete(name=pdf_file.name)
        except Exception: pass
        return None

    try:
        project_data = json.loads(raw1)
    except json.JSONDecodeError as e:
        logging.warning(f"  Pass 1 JSON truncated or invalid: {e} — attempting auto-repair ...")
        repaired1 = repair_truncated_json(raw1)
        try:
            project_data = json.loads(repaired1)
            logging.info("  Pass 1 auto-repair succeeded.")
        except json.JSONDecodeError as e2:
            logging.error(f"  Pass 1 JSON repair failed: {e2}")
            try: client.files.delete(name=pdf_file.name)
            except Exception: pass
            print(f"\n\n--- RAW INVALID JSON PASS 1 ---\n{raw1}\n------------------------------\n\n")
            return None

    unit_names = project_data.pop("unit_names", [])
    logging.info(f"  Found {len(unit_names)} unit(s): {unit_names}")

    if not unit_names:
        logging.warning("  No unit names found — brochure may lack floor plans.")
        try: client.files.delete(name=pdf_file.name)
        except Exception: pass
        project_data["units"] = []
        return project_data

    # ── Pass 2: extract each unit individually with schema ────────────────
    units = []
    for i, unit_name in enumerate(unit_names):
        logging.info(f"  Pass 2 [{i+1}/{len(unit_names)}]: extracting '{unit_name}' ...")
        raw2 = call_gemini(
            client,
            contents=[pdf_file, make_pass2_prompt(unit_name)],
            schema=UnitSchema,
            max_tokens=8192,   # one unit at a time — always fits
        )
        if not raw2:
            logging.warning(f"  Failed to extract '{unit_name}' — skipping.")
            continue
        try:
            unit_data = json.loads(raw2)
            units.append(unit_data)
        except json.JSONDecodeError as e:
            logging.warning(f"  JSON error for '{unit_name}': {e} — skipping.")
            continue

    # Cleanup remote file
    try:
        client.files.delete(name=pdf_file.name)
    except Exception:
        pass

    project_data["units"] = units
    return project_data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Extract JSON from real estate brochures using Gemini.")
    p.add_argument("--no-preprocess",          action="store_true")
    p.add_argument("--dpi",     type=int, default=250)
    p.add_argument("--quality", type=int, default=82)
    p.add_argument("--overwrite-preprocessed", action="store_true")
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
                logging.info(f"  Preprocessed PDF -> {work_path}")
            except Exception as e:
                logging.error(f"  Preprocessing failed: {e} — using raw PDF.")
                work_path = pdf_path
        else:
            work_path = pdf_path

        # Step 2: Two-pass extraction
        data = process_pdf(client, work_path)

        if data is None:
            logging.error(f"  SKIPPED — extraction failed for {pdf_path.name}")
            continue

        # Step 3: Postprocess
        data = clean_null_strings(data)
        data["project_id"] = str(uuid.uuid4())
        data["brochure_file"] = pdf_path.name
        data["units"] = fix_hallucinated_dimensions(data.get("units", []))
        data["units"] = compute_room_areas(data.get("units", []))

        # Sanity: impossible SBA
        for unit in data.get("units", []):
            bhk = unit.get("bhk") or 0
            sba = unit.get("super_built_up_area_sqft")
            ptype = unit.get("property_type", "")
            utype = unit.get("unit_type", "?")
            if sba and bhk >= 4 and ptype in ("VILLA", "ROW_HOUSE", "TENEMENT") and sba < 1500:
                logging.warning(f"  [SANITY] SBA={sba} too small for {bhk}BHK '{utype}'. Nulling.")
                unit["super_built_up_area_sqft"] = None
            for room in unit.get("rooms", []):
                if room.get("area_sqft") == 0:
                    room["area_sqft"] = None

        out_file = args.output_dir / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"  SAVED -> {out_file}  ({len(data.get('units', []))} units)")
        success += 1

    logging.info(f"\nDone! {success}/{len(pdf_files)} extracted successfully.")


if __name__ == "__main__":
    main()