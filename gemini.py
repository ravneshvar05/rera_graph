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
# Note: project_id is intentionally EXCLUDED from BrochureData so the LLM
# never sees or touches it. We inject it ourselves in postprocess().
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
    name: str = Field(description="Normalized common room label (e.g., 'Hall' instead of 'Drawing Room', 'Balcony' instead of 'Balc'). Use original name if not a common room type.")
    length: Optional[str] = Field(default=None, description="Raw length string exactly as written anywhere on the page (floor plan drawing OR side legend/table). DO NOT include 'x' or 'X'. DO NOT guess from single numbers, sequence IDs, or areas. Set to null if no length x width pair exists for this room.")
    width:  Optional[str] = Field(default=None, description="Raw width string exactly as written anywhere on the page (floor plan drawing OR side legend/table). DO NOT include 'x' or 'X'. DO NOT guess from single numbers, sequence IDs, or areas. Set to null if no length x width pair exists for this room.")
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
    description: Optional[str] = Field(default=None, description="A rich, informative paragraph describing the overall vibe, environment, lifestyle, layout, and architectural style of the society. This is used for semantic/vector search, so include keywords about the surroundings and atmosphere instead of leaving it null.")
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
    description: Optional[str] = Field(default=None, description="A comprehensive 1-2 sentence description of the unit capturing its vibe, unique architectural aspects, layout (e.g. open-concept, courtyard, spacious), or special features. This is critical for vector search, so synthesize a good description instead of leaving it blank.")
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

# FIX: project_id removed — LLM never sees it, we inject UUID ourselves
class BrochureData(BaseModel):
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
# PROMPT
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = """
You are a real estate data extraction AI. Analyze this ENTIRE brochure PDF (every page) and extract ALL data into the JSON schema.

RULES:

1. UNITS (most important):
   - Extract EVERY unit/variant (1A, 1B, 1C...) as a SEPARATE unit with its OWN rooms and dimensions.
   - Never copy dimensions from one variant to another. Read each floor plan independently.
   - Scan every page — do NOT stop early.
   - ONLY extract rooms that are explicitly drawn or described for that specific unit. DO NOT hallucinate standard rooms (like a Bedroom or Balcony) if they do not exist in the floor plan.
   - IMPORTANT: If a room (e.g. "Bedroom") does NOT exist in the unit (e.g. it is a "1 HK"), omit it entirely from the `rooms` list. DO NOT output a room object with null values just to satisfy perceived schema requirements.
   - If a room IS genuinely present in the unit but lacks printed dimensions, you MUST extract its name and set its length, width, and area to null.
   - CRITICAL: While dimensions are usually drawn directly inside the room, SOMETIMES they are listed in a legend/table ON THE SIDE instead (e.g. the plan shows 'A' or '1' inside the room, and the side text says 'A. LIVING (17\\' X 11\\')'). If you see letters or numbers instead of dimensions in the rooms, you MUST actively scan the entire page, including all side texts and legends, to find and match the corresponding room names and dimensions. DO NOT STOP SEARCHING.

2. ROOM NAMES — use these canonical names in the `name` field:
   - Drawing Room / Living Room / Lounge / Family Room / L-D / Drg Room → "Hall"
   - Mast Bed / M. Bedroom / Master Bed / M.B.R → "Master Bedroom"
   - Bed / Bed Room / BED ROOM → "Bedroom"
   - Dining Room / Dinning / Dinning Room → "Dining"
   - Toi / W.C / WC / Powder Room → "Toilet"
   - Balc / Deck / Verandah → "Balcony"
   - Wash / W.A / Utility (wash area context) → "Wash Area"
   - Puja Room / Prayer Room / Mandir / Puja / Pooja → "Pooja Room"
   - Storage Room / Store / Storeroom → "Store Room"
   - Study → "Study Room"
   - Maid Room / Maid / Servant → "Servant Room"
   - Bath Room → "Bathroom"
   - Dress Room / Wardrobe Room → "Dressing Room"
   - Foyer / Entrance (lobby context) → "Lobby"
   - Corridor / Common Passage → "Passage"
   - For anything else, use the brochure's exact printed name.

3. DIMENSIONS (STRICTLY PAIRED ONLY):
   - ONLY extract dimensions if explicitly printed as a length and width pair (e.g., "10'0\" x 11'4\"" or "3.05 x 3.45").
   - ALWAYS split the pair at the 'x' or 'X'. Extract the first part into `length` and the second part into `width`.
   - NEVER put the entire "10'2\" x 12'0\"" string into a single field.
   - Extract length and width into their SEPARATE fields. NEVER include 'x' or 'X' in the value.
   - Example printed text "12' x 10'1\"" → length="12'", width="10'1\"" (NOT "12'x").
   - Example printed text "10'2\"X12\"" → length="10'2\"", width="12\"".
   - NEVER hallucinate dimensions from single numbers (e.g., room counts, sequence numbers, or total sqft areas).
   - If a room does NOT have an explicitly printed Length x Width pair on the floor plan or legend, you MUST leave both `length` and `width` as `null`. NEVER GUESS.
   - Sanity check: Master Bedroom > Bedroom > Kitchen ≈ Dining > Toilet (toilet width ≤ 5').
   - CRITICAL: If a room's name and dimensions are listed in a side legend (like "A. LIVING & DINING (17' X 11')"), you MUST extract "17'" as length and "11'" as width for that specific room object. NEVER output null dimensions when a legend exists.
   - CRITICAL PENTHOUSE RULE: For large units like "5 BHK Penthouse" or "4 BHK", the dimensions are ALMOST ALWAYS in a side table or list, and NOT DRAWN on the room. You MUST NOT LEAVE DIMENSIONS NULL for large units just because the drawing is complex. Search the entire page for the legend/table containing the dimensions.

4. ROOM AREA: Use printed value if available. Otherwise compute area_sqft = length_ft × width_ft (1m = 3.281ft).

5. MISSING DATA & TOTAL AREAS:
   - JSON null only (never "null", "NA", "None").
   - EXTRACT TOTAL AREAS (super_built_up_area_sqft, carpet_area_sqft) wherever printed (tables, unit titles, floor plans).
   - ALWAYS convert Sq. Yard / Sq. Yds to Sq. Ft. (1 Sq. Yard = 9 Sq. Ft.). Example: "100 sq yds" -> 900.
   - If the area is genuinely not printed anywhere for the unit, use null. Do NOT estimate it yourself.

6. ROOM CONNECTIONS (VERIFY CAREFULLY — LLMs commonly make mistakes here):
   For EACH bedroom, follow these steps:
   a) Look at every wall of that bedroom in the floor plan.
   b) Is there a door from that bedroom wall directly into a Toilet or Bathroom? → attached_bathroom = true
   c) IMPORTANT: A Toilet that can ONLY be entered from ONE specific bedroom is always "attached". A common/shared toilet entered from the passage/hall is NOT attached to any bedroom.
   d) Is there a balcony directly joined to that bedroom (sliding door or opening)? → has_balcony_access = true
   - Default to false only after completing steps a-d. Never default to false without checking.
   - A faintly drawn door still counts. A shared common toilet does NOT count as attached.

7. ENTRANCE FACING: Trace compass to find which direction main door faces. null only if NO compass exists anywhere.

8. VILLAS: applicable_buildings = []. No floor layout data needed.

9. LANDMARKS: Real landmarks only (schools, hospitals, malls, transit). Correct OCR errors ("ISION TEMPLE" → "ISKCON TEMPLE").

10. AMENITIES: List every amenity mentioned anywhere in the brochure.

11. DESCRIPTIONS (CRITICAL): Synthesize informative, rich descriptions for both `SocietyLayoutSchema.description` and `UnitSchema.description`. Capture the vibe, lifestyle, architecture, layout, and any special features. This is vital for semantic search matching. DO NOT leave these fields null if you can infer details from the imagery or text.

Output strictly as JSON matching the schema.
"""


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def repair_truncated_json(text: str) -> str:
    """
    If Gemini cuts off mid-JSON, close all open brackets/braces so the
    output is at least parseable.
    """
    text = text.strip()
    if text.startswith("```json"): text = text[7:].strip()
    if text.endswith("```"): text = text[:-3].strip()
    text = re.sub(r",\s*$", "", text)
    
    stack, in_string, escape = [], False, False
    for ch in text:
        if escape: escape = False; continue
        if ch == "\\" and in_string: escape = True; continue
        if ch == '"': in_string = not in_string; continue
        if in_string: continue
        if ch in "{[": stack.append(ch)
        elif ch in "}]" and stack: stack.pop()
        
    if in_string: text += '"'
    closing = {"[": "]", "{": "}"}
    for opener in reversed(stack): text += closing[opener]
    return text

# FIX 1: clean "null" strings, "NA", "None" → Python None recursively
_NULL_STRINGS = {"null", "na", "none", "n/a", "nil", "undefined", "-"}

def clean_null_strings(obj):
    """Recursively replace null-like strings with None throughout the dict."""
    if isinstance(obj, dict):
        return {k: clean_null_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_null_strings(i) for i in obj]
    if isinstance(obj, str) and obj.strip().lower() in _NULL_STRINGS:
        return None
    return obj


def parse_ft(raw: str) -> Optional[float]:
    """Parse a raw dimension string into feet (float)."""
    if not raw:
        return None
    raw = raw.strip()
    # Remove any stray "x" or "X" at the end if LLM messed up
    raw = re.sub(r'[\sxX]+$', '', raw)
    
    # 13.6' or 10.5'-6" or 10'6" or 10'
    m = re.match(r"(\d+(?:\.\d+)?)['\u2019](?:[\s\-]?(\d+(?:\.\d+)?)[\"\u201d]?)?", raw)
    if m:
        ft = float(m.group(1))
        inch = float(m.group(2)) if m.group(2) else 0.0
        return ft + (inch / 12.0)
        
    # Standalone inches: 150"
    m = re.match(r"^(\d+(?:\.\d+)?)[\"\u201d]$", raw)
    if m:
        return float(m.group(1)) / 12.0

    # 10-6 (feet-inches with dash)
    m = re.match(r"^(\d+)-(\d+)$", raw)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 12.0
        
    # Plain decimal with m, mt, mtr, or meters
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:m|mtr|mt|meters?)$", raw, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 3.281
        
    # Plain decimal with ft or feet
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:ft|feet)$", raw, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Plain decimal — metres if < 15, else feet
    m = re.match(r"^(\d+\.?\d*)$", raw)
    if m:
        val = float(m.group(1))
        return val * 3.281 if val < 15 else val
    return None


def compute_room_areas(data: dict) -> dict:
    """Fill area_sqft for rooms that have length + width but no explicit area."""
    for unit in data.get("units", []):
        for room in unit.get("rooms", []):
            if room.get("area_sqft") is not None:
                continue
            l = parse_ft(room.get("length") or "")
            w = parse_ft(room.get("width") or "")
            if l and w:
                room["area_sqft"] = round(l * w, 1)
    return data



# ── Room name normalisation (deterministic, runs AFTER LLM extraction) ────────
#
# Maps lowercase substrings → canonical room name.
# Checked IN ORDER — first match wins. More-specific patterns listed first.
# This guarantees consistent names in the JSON regardless of LLM output.

_ROOM_NORM: list[tuple[str, str]] = [
    # Master Bedroom — check BEFORE generic "bedroom"/"bed"
    ("master bed",     "Master Bedroom"),
    ("m.bed",          "Master Bedroom"),
    ("m bed",          "Master Bedroom"),
    ("m.b.r",          "Master Bedroom"),
    ("mbr",            "Master Bedroom"),
    # Hall / Drawing Room variants
    ("drawing room",   "Hall"),
    ("living room",    "Hall"),
    ("family room",    "Hall"),
    ("drg room",       "Hall"),
    ("dr room",        "Hall"),
    ("lounge",         "Hall"),
    ("l/d",            "Hall"),
    ("l-d",            "Hall"),
    # Bedroom — after master bedroom checks
    ("bed room",       "Bedroom"),
    # Dining
    ("dining room",    "Dining"),
    ("dinning room",   "Dining"),
    ("dinning",        "Dining"),
    # Toilet / WC
    ("powder room",    "Toilet"),
    ("w.c",            "Toilet"),
    # Balcony
    ("verandah",       "Balcony"),
    ("balc",           "Balcony"),
    ("deck",           "Balcony"),
    # Wash Area — "wash area" already correct, catch abbrevs
    ("utility",        "Wash Area"),
    ("laundry",        "Wash Area"),
    ("w.a",            "Wash Area"),
    # Pooja Room variants
    ("puja room",      "Pooja Room"),
    ("prayer room",    "Pooja Room"),
    ("mandir",         "Pooja Room"),
    # Store Room
    ("storage room",   "Store Room"),
    ("storeroom",      "Store Room"),
    # Study Room
    ("study room",     "Study Room"),   # already canonical — keep for case normalisation
    # Servant Room
    ("servant room",   "Servant Room"), # already canonical — keep for case normalisation
    ("maid room",      "Servant Room"),
    # Bathroom
    ("bath room",      "Bathroom"),
    # Dressing Room
    ("dressing room",  "Dressing Room"),# already canonical
    ("dress room",     "Dressing Room"),
    ("wardrobe room",  "Dressing Room"),
    # Lobby
    ("foyer",          "Lobby"),
    # Passage
    ("corridor",       "Passage"),
    ("common passage", "Passage"),
    # Generic single-word catches — keep LAST to avoid over-matching
    ("wash",           "Wash Area"),
    ("puja",           "Pooja Room"),
    ("pooja",          "Pooja Room"),
    ("store",          "Store Room"),
    ("study",          "Study Room"),
    ("servant",        "Servant Room"),
    ("maid",           "Servant Room"),
    ("bed",            "Bedroom"),
]


def normalize_room_names(data: dict) -> dict:
    """
    Post-process: apply canonical room name normalisation to every room in every unit.
    Uses substring matching (case-insensitive) so abbreviations, typos, and
    mixed-case variants are all caught.  First match in _ROOM_NORM wins.
    """
    for unit in data.get("units", []):
        for room in unit.get("rooms", []):
            raw = (room.get("name") or "").strip()
            if not raw:
                continue
            lower = raw.lower()
            for fragment, canonical in _ROOM_NORM:
                if fragment in lower:
                    if raw != canonical:
                        logging.debug(f"  [NORM] room '{raw}' → '{canonical}'")
                        room["name"] = canonical
                    break   # first match wins
    return data


def postprocess(data: dict, filename: str) -> dict:
    """Apply all fixes and return cleaned data with injected project_id."""

    # FIX 1: clean "null" strings everywhere
    data = clean_null_strings(data)

    # FIX 2: inject project_id ourselves (never trust LLM for this)
    data["project_id"] = str(uuid.uuid4())

    # Always stamp correct filename
    data["brochure_file"] = filename

    # Compute room areas from dimensions
    data = compute_room_areas(data)

    # Unit area estimation removed per user request.

    # Normalize room names to canonical form (deterministic, after LLM)
    data = normalize_room_names(data)

    # Sanity: impossible SBA values
    for unit in data.get("units", []):
        bhk   = unit.get("bhk") or 0
        sba   = unit.get("super_built_up_area_sqft")
        ptype = unit.get("property_type", "")
        utype = unit.get("unit_type", "?")
        if sba and bhk >= 4 and ptype in ("VILLA", "ROW_HOUSE", "TENEMENT") and sba < 1500:
            logging.warning(f"  [SANITY] SBA={sba} sqft too small for {bhk}BHK {ptype} '{utype}'. Nulling.")
            unit["super_built_up_area_sqft"] = None
        for room in unit.get("rooms", []):
            if room.get("area_sqft") == 0:
                room["area_sqft"] = None

    return data




# ══════════════════════════════════════════════════════════════════════════════
# CORE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(client_factory: callable, pdf_path: Path, max_retries: int = 3) -> str | None:
    logging.info(f"  Uploading: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)")

    for attempt in range(max_retries):
        try:
            client, key_idx = client_factory()
            logging.info(f"  Attempt {attempt + 1}: Using API Key #{key_idx + 1}")
            
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
                    max_output_tokens=65000,
                ),
            )

            try:
                client.files.delete(name=pdf_file.name)
            except Exception:
                pass

            return response.text

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Resource Exhausted" in err_msg or "quota" in err_msg.lower():
                logging.warning(f"  Rate limit hit (429) on attempt {attempt + 1}. Error: {str(e)[:150]}")
                if attempt < max_retries - 1:
                    logging.info("  Retrying with next API key...")
                    time.sleep(2)
                    continue
                else:
                    logging.error("  Max retries reached on rate limit.")
            else:
                 logging.error(f"  API call failed: {err_msg[:400]}")
                 return None
                 
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Extract JSON from real estate brochures using Gemini.")
    p.add_argument("--no-preprocess",            action="store_true", help="Skip PDF preprocessing")
    p.add_argument("--dpi",      type=int,  default=250,             help="Preprocessing DPI (default: 250)")
    p.add_argument("--quality",  type=int,  default=82,              help="JPEG quality (default: 82)")
    p.add_argument("--overwrite-preprocessed",   action="store_true", help="Re-preprocess existing files")
    p.add_argument("--input-dir",  type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--prep-dir",   type=Path, default=Path("preprocessed"))
    return p.parse_args()


def main():
    args = parse_args()

    api_key_1 = os.getenv("GEMINI_API_KEY")
    api_key_2 = os.getenv("GEMINI_API_KEY_2")
    
    api_keys = [k for k in [api_key_1, api_key_2] if k]

    if not api_keys:
        logging.error("No valid GEMINI_API_KEY found in .env file.")
        return

    # Simple round-robin key tracking
    current_key_idx = [0]
    
    def get_client() -> tuple[genai.Client, int]:
        idx = current_key_idx[0] % len(api_keys)
        current_key_idx[0] += 1
        return genai.Client(api_key=api_keys[idx]), idx
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
                logging.info(f"  Preprocessed PDF -> {work_path}  (open to verify quality)")
            except Exception as e:
                logging.error(f"  Preprocessing failed: {e} — using raw PDF.")
                work_path = pdf_path
        else:
            work_path = pdf_path

        # Step 2: Extract
        logging.info("  Step 2: Extracting ...")
        result_text = process_pdf(get_client, work_path)

        if not result_text:
            logging.error(f"  SKIPPED — no result for {pdf_path.name}")
            continue

        # Step 3: Parse and Postprocess
        data = None
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            logging.warning("  JSON truncated — attempting auto-repair ...")
            repaired = repair_truncated_json(result_text)
            try:
                data = json.loads(repaired)
                logging.info("  Auto-repair succeeded.")
            except json.JSONDecodeError as e:
                logging.error(f"  Repair failed: {e} — saving raw output.")
                final_json = result_text

        if data is not None:
            data = postprocess(data, pdf_path.name)
            final_json = json.dumps(data, ensure_ascii=False, indent=2)

        out_file = args.output_dir / f"{pdf_path.stem}.json"
        out_file.write_text(final_json, encoding="utf-8")
        logging.info(f"  SAVED -> {out_file}")
        success += 1

    logging.info(f"\nDone! {success}/{len(pdf_files)} extracted successfully.")


if __name__ == "__main__":
    main()