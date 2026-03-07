import os
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Setup Logging
Path("pdf_tests/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_tests/logs/gemini_pdf_extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ==========================================
# 1. THE PERFECTED BROCHURE SCHEMA
# ==========================================

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

from typing import Optional, Literal, Union

class BaseRoomSchema(BaseModel):
    name: str = Field(description="Exact name on the plan, e.g., 'Master Bedroom' or 'Kitchen'")
    length: Optional[str] = Field(default=None, description="Raw length string exactly as written, e.g., '10-0' or '10\\'0\"'")
    width: Optional[str] = Field(default=None, description="Raw width string exactly as written, e.g., '11-6' or '11\\'6\"'")
    area_sqft: Optional[float] = None
    floor_level: Optional[str] = Field(default=None, description="e.g., Ground, First. Vital for tenements/villas.")

class PrimaryRoomSchema(BaseRoomSchema):
    room_type: Literal[RoomType.BEDROOM, RoomType.DRAWING_ROOM]
    attached_bathroom: Optional[bool] = Field(default=None, description="True if a bathroom is directly connected to this bedroom")
    has_balcony_access: Optional[bool] = Field(default=None, description="True if a balcony is directly connected to this room")

class StandardRoomSchema(BaseRoomSchema):
    room_type: Literal[
        RoomType.KITCHEN, RoomType.DINING, RoomType.TOILET, RoomType.BATHROOM, 
        RoomType.WC, RoomType.BALCONY, RoomType.TERRACE, RoomType.WASH_AREA, 
        RoomType.POOJA_ROOM, RoomType.STORE_ROOM, RoomType.STUDY_ROOM, 
        RoomType.SERVANT_ROOM, RoomType.UTILITY_ROOM, RoomType.PASSAGE, 
        RoomType.LOBBY, RoomType.FOYER, RoomType.DRESSING_ROOM, RoomType.COURTYARD, RoomType.OTHER
    ] = RoomType.OTHER

class SocietyLayoutSchema(BaseModel):
    description: Optional[str] = None
    total_apartment_blocks: Optional[int] = Field(default=None, description="Total number of towers/blocks. ONLY FOR APARTMENTS. For Villas/Row Houses, leave this as null.")
    total_independent_villas_or_tenements: Optional[int] = Field(default=None, description="Total number of independent ground-level houses. ONLY FOR VILLAS/TENEMENTS/ROW_HOUSES. For Apartments, leave this as null.")
    building_names: list[str] = Field(default_factory=list, description="For apartments, list tower/block names (e.g. 'Tower A'). For Villas/Row Houses, leave this empty, do NOT list individual house numbers.")
    has_clubhouse: Optional[bool] = None
    has_park_or_garden: Optional[bool] = None
    has_swimming_pool: Optional[bool] = None
    has_sports_courts: Optional[bool] = None
    has_parking_area: Optional[bool] = None
    commercial_shops_included: Optional[bool] = Field(default=None, description="True if it's a mixed-use society with shops")
    road_width_details: list[str] = Field(default_factory=list)

class FloorLayoutSchema(BaseModel):
    layout_name: Optional[str] = Field(default=None, description="e.g., 'Typical Floor Plan 1st to 10th'. ONLY FOR APARTMENTS. Leave empty for Villas / Row Houses.")
    total_units_on_floor: Optional[int] = None
    has_lifts: Optional[bool] = None
    has_staircases: Optional[bool] = None
    corridor_width: Optional[str] = None
    has_refuge_area: Optional[bool] = Field(default=None)

class UnitSchema(BaseModel):
    unit_type: str = Field(description="e.g. '2 BHK', '3 BHK + Maid', '4 BHK Villa'")
    property_type: PropertyType = Field(default=PropertyType.OTHER, description="The architectural type of the property")
    applicable_buildings: list[str] = Field(default_factory=list, description="Which towers/blocks this applies to. If the property type is VILLA or ROW_HOUSE, leave this completely empty.")
    entrance_Facing: Optional[str] = Field(default=None, description="Vastu direction the main door faces, e.g., East, North-East. If you cannot find it explicitly, set to null. DO NOT hallucinate, as this will be reviewed by a human.")
    description: Optional[str] = Field(default=None, description="Descriptive text capturing the vibe, architecture, special features, or layout unique to this unit. Extremely useful for Vector Search.")
    
    bhk: Optional[int] = None
    carpet_area_sqft: Optional[float] = None
    balcony_area_sqft: Optional[float] = Field(default=None, description="Detailed separate balcony area if listed")
    wash_area_sqft: Optional[float] = Field(default=None, description="Detailed separate wash area if listed")
    super_built_up_area_sqft: Optional[float] = Field(default=None, description="Also known as saleable area")
    
    rooms: list[Union[PrimaryRoomSchema, StandardRoomSchema]] = Field(default_factory=list)

class LocationSchema(BaseModel):
    city: Optional[str] = None
    neighbourhood: Optional[str] = None
    address: Optional[str] = None
    pin_code: Optional[str] = None
    nearby_landmarks: list[str] = Field(default_factory=list, description="Schools, hospitals, transit hubs mentioned")

class BrochureData(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    brochure_file: str = Field(description="The filename of the processed brochure")
    project_name: Optional[str] = None
    developer_name: Optional[str] = Field(default=None)
    rera_registration_number: Optional[str] = Field(default=None)
    project_status: ProjectStatus = Field(default=ProjectStatus.UNKNOWN)
    possession_date: Optional[str] = Field(default=None)
    
    location: LocationSchema = Field(default_factory=LocationSchema)
    amenities: list[str] = Field(default_factory=list)
    society_layout: Optional[SocietyLayoutSchema] = None
    floor_layouts: list[FloorLayoutSchema] = Field(default_factory=list)
    units: list[UnitSchema] = Field(default_factory=list)


# ==========================================
# 2. PROMPT ENGINEERING & EXECUTION LOGIC
# ==========================================

MODEL = "gemini-2.5-flash"

# Number of PDFs to process in parallel.
# 4 is conservative and stays within Gemini Flash rate limits.
# Increase to 8 if you have a higher-tier API quota.
MAX_WORKERS = 4

# A highly specific prompt to guide the LLM through complex real estate nuances
PROMPT = """
You are an expert real estate architect and data extraction AI. 
Analyze this entire real estate brochure PDF and extract all project data, structural details, and unit floor plans.

CRITICAL INSTRUCTIONS:
1. DIMENSIONS: Real estate dimensions are messy. Extract the length and width EXACTLY as written in the text (e.g., "10'0\" x 11'4\"", "3.05 x 3.45"). Do NOT attempt to convert these strings into floats. If a dimension includes both metric and imperial, extract the first one presented.
2. MISSING DATA: If a field is not present, use the exact JSON literal `null`. Do NOT write the string "null". Every null must be a true JSON null without quotes.
3. VILLA vs APARTMENT LOGIC: Floor Layouts (corridors, lifts) are ONLY for apartments. Do NOT map individual villa/house plot numbers as "buildings". For villas, the unit `applicable_buildings` should be empty or just the project name.
4. IMPLICIT DATA: Look at the visual floor plans. If a bathroom has a door directly into a BEDROOM or DRAWING_ROOM, set 'attached_bathroom' to true. If a balcony is attached to a BEDROOM or DRAWING_ROOM, set 'has_balcony_access' to true. ONLY bedrooms and drawing rooms have these fields.
5. PROPERTY TYPES: For the `property_type` on units, classify as VILLA for detached houses, ROW_HOUSE for joined row houses, TENEMENT for independent ground-level houses, and APARTMENT for flats.
6. VASTU & ENTRANCE: Look carefully for compass symbols (N, S, E, W) and use them to visually deduce the entrance facing direction of each unit. For APARTMENTS, trace the door facing the lobby. For VILLAS/TENEMENTS, trace the main entrance door from the street/plot. Explicit text like "East-facing" may not be present, so you MUST visually trace it using the compass. Capture it in the `entrance_Facing` field as a string (e.g. "East"). Only set to the absolute JSON `null` (without quotes) if the entire brochure completely lacks both explicit text AND a compass symbol. DO NOT guess without visual evidence.
7. DESCRIPTIONS: For the `description` on units, synthesize a short sentence capturing its vibe, architecture, or special features (e.g., "Spacious open-concept layout with large courtyard").

Output strictly as a JSON object matching the requested schema.
"""

def process_pdf(client: genai.Client, pdf_path: Path) -> str | None:
    logging.info(f"  Uploading PDF: {pdf_path.name}")
    
    try:
        pdf_file = client.files.upload(file=str(pdf_path), config={'mime_type': 'application/pdf'})
        
        logging.info("  Processing PDF on server...")
        while pdf_file.state.name == 'PROCESSING':
            print(".", end="", flush=True)
            time.sleep(2)
            pdf_file = client.files.get(name=pdf_file.name)
        print() 

        if pdf_file.state.name == 'FAILED':
            logging.error("  File processing failed on the server.")
            return None

        logging.info(f"  Calling {MODEL}...")
        
        response = client.models.generate_content(
            model=MODEL,
            contents=[pdf_file, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BrochureData, # Using the finalized comprehensive schema
                temperature=0.0, 
            )
        )
        
        client.files.delete(name=pdf_file.name)
        return response.text
        
    except Exception as e:
        logging.error(f"  API call failed: {str(e)[:300]}")
        return None

def _process_one(api_key: str, pdf_path: Path, output_dir: Path, idx: int, total: int) -> bool:
    """
    Worker function called by each thread.
    Creates its own Gemini client (thread-safe — each thread has its own HTTP session).
    Returns True on success, False on failure.
    """
    logging.info(f"--- [{idx}/{total}] Processing: {pdf_path.name} ---")
    try:
        client = genai.Client(api_key=api_key)
        result = process_pdf(client, pdf_path)
        if result:
            out_file = output_dir / f"{pdf_path.stem}.json"
            out_file.write_text(result, encoding="utf-8")
            logging.info(f"  SAVED -> {out_file}")
            return True
        else:
            logging.error(f"  SKIPPED (empty response): {pdf_path.name}")
            return False
    except Exception as e:
        logging.error(f"  FAILED: {pdf_path.name} — {e}")
        return False


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not found in .env file.")
        return

    input_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip PDFs that already have a JSON output (idempotent re-runs)
    pdf_files = [
        p for p in sorted(input_dir.glob("*.pdf"))
        if not (output_dir / f"{p.stem}.json").exists()
    ]

    if not pdf_files:
        logging.info("No new PDFs to process (all already have JSON output).")
        return

    total = len(pdf_files)
    logging.info(f"Found {total} new PDF(s) to process (workers={MAX_WORKERS}).")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all PDFs; track future → pdf_path for error reporting
        future_to_pdf = {
            executor.submit(_process_one, api_key, pdf_path, output_dir, idx + 1, total): pdf_path
            for idx, pdf_path in enumerate(pdf_files)
        }

        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                logging.error(f"  Unhandled error for {pdf_path.name}: {exc}")
                fail_count += 1

    logging.info(f"Done! {success_count}/{total} saved, {fail_count} failed.")

if __name__ == "__main__":
    main()