"""
preprocess.py  —  PDF Enhancement Pipeline
===========================================
Standalone script to preprocess real estate brochure PDFs before Gemini extraction.

Usage:
    python preprocess.py                    # processes all PDFs in ./data/
    python preprocess.py myfile.pdf         # processes a single PDF
    python preprocess.py --dpi 300          # override DPI (default: 250)
    python preprocess.py --quality 85       # override JPEG quality (default: 82)
    python preprocess.py --no-enhance       # skip contrast/sharpness enhancement

Output:
    Preprocessed PDFs are saved to ./preprocessed/
    You can open them and verify quality before running gemini.py

Install dependencies:
    pip install pymupdf pillow
"""

import io
import sys
import argparse
import logging
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import fitz
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    print("ERROR: Missing dependencies. Please run:")
    print("    pip install pymupdf pillow")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
Path("pdf_tests/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_tests/logs/preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DPI      = 250    # 250 DPI: sharp text, readable dims, stays < 15 MB
DEFAULT_QUALITY  = 82     # JPEG quality: good balance of size vs clarity
MAX_MB           = 15     # if output exceeds this, auto-reduce quality


# ══════════════════════════════════════════════════════════════════════════════
# Core preprocessing function
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int       = DEFAULT_DPI,
    quality: int   = DEFAULT_QUALITY,
    enhance: bool  = True,
    overwrite: bool = False,
) -> Path:
    """
    Render each PDF page at `dpi` DPI, apply enhancement pipeline,
    then repack all pages into a new lean PDF.

    Enhancement pipeline (when enhance=True):
        1. UnsharpMask   — recovers blurry/faint dimension text
        2. Contrast +25% — helps very faint text become readable
        3. Brightness +5% — lifts dark/muddy scans slightly

    Returns:
        Path to the preprocessed PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}_preprocessed.pdf"

    if out_path.exists() and not overwrite:
        logging.info(f"  Already exists (use --overwrite to redo): {out_path.name}")
        return out_path

    logging.info(f"  Input  : {pdf_path}  ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    logging.info(f"  DPI    : {dpi}  |  JPEG quality: {quality}  |  Enhance: {enhance}")

    doc_in  = fitz.open(str(pdf_path))
    doc_out = fitz.open()
    mat     = fitz.Matrix(dpi / 72, dpi / 72)   # 72 DPI is the PDF baseline

    for page_num, page in enumerate(doc_in):
        # ── Render page to RGB pixels ─────────────────────────────────────
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if enhance:
            # 1. Sharpen  — UnsharpMask is gentler than SHARPEN filter
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
            # 2. Contrast boost
            img = ImageEnhance.Contrast(img).enhance(1.25)
            # 3. Slight brightness lift
            img = ImageEnhance.Brightness(img).enhance(1.05)

        # ── Encode as JPEG in memory ──────────────────────────────────────
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)

        # ── Insert as a new page at the same physical dimensions ──────────
        new_page = doc_out.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), stream=buf.read())

        if (page_num + 1) % 5 == 0 or (page_num + 1) == len(doc_in):
            logging.info(f"    Page {page_num + 1}/{len(doc_in)} done")

    doc_out.save(str(out_path), deflate=True, garbage=4)
    doc_in.close()
    doc_out.close()

    size_mb = out_path.stat().st_size / 1_048_576
    logging.info(f"  Output : {out_path}  ({size_mb:.1f} MB)")

    # ── Auto-reduce if file is too large for Gemini ───────────────────────
    if size_mb > MAX_MB:
        logging.warning(f"  File is {size_mb:.1f} MB > {MAX_MB} MB limit — reducing quality ...")
        out_path = _reduce_quality(out_path, output_dir)

    return out_path


def _reduce_quality(pdf_path: Path, output_dir: Path) -> Path:
    """Re-encode at q=65 to bring an oversized PDF under the size limit."""
    out_path = output_dir / f"{pdf_path.stem}_reduced.pdf"
    doc_in   = fitz.open(str(pdf_path))
    doc_out  = fitz.open()

    for page in doc_in:
        pix = page.get_pixmap(colorspace=fitz.csRGB, alpha=False)
        buf = io.BytesIO()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(buf, format="JPEG", quality=65, optimize=True)
        buf.seek(0)
        new_page = doc_out.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), stream=buf.read())

    doc_out.save(str(out_path), deflate=True, garbage=4)
    doc_in.close()
    doc_out.close()

    size_mb = out_path.stat().st_size / 1_048_576
    logging.info(f"  Reduced: {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess real estate brochure PDFs for better Gemini extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess.py                        # all PDFs in ./data/
  python preprocess.py KP_Villas.pdf          # single file (auto-finds in ./data/)
  python preprocess.py --dpi 300              # higher DPI for very blurry brochures
  python preprocess.py --quality 90           # higher JPEG quality
  python preprocess.py --no-enhance           # skip sharpening/contrast
  python preprocess.py --overwrite            # redo already-processed files
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="PDF file(s) to process. If omitted, processes all PDFs in ./data/",
    )
    parser.add_argument("--dpi",       type=int,  default=DEFAULT_DPI,     help=f"Render DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--quality",   type=int,  default=DEFAULT_QUALITY, help=f"JPEG quality 1-95 (default: {DEFAULT_QUALITY})")
    parser.add_argument("--no-enhance",action="store_true",                help="Skip contrast/sharpness enhancement")
    parser.add_argument("--overwrite", action="store_true",                help="Overwrite existing preprocessed files")
    parser.add_argument("--input-dir", type=Path, default=Path("data"),    help="Input directory (default: ./data/)")
    parser.add_argument("--output-dir",type=Path, default=Path("preprocessed"), help="Output directory (default: ./preprocessed/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve input files
    if args.files:
        pdf_paths = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                # Try finding it in the input dir
                p = args.input_dir / f
            if not p.exists():
                logging.error(f"File not found: {f}")
                continue
            pdf_paths.append(p)
    else:
        pdf_paths = sorted(args.input_dir.glob("*.pdf"))
        if not pdf_paths:
            logging.error(f"No PDFs found in '{args.input_dir}'. Use --input-dir to specify a different folder.")
            sys.exit(1)

    logging.info(f"Found {len(pdf_paths)} PDF(s) to preprocess.")
    logging.info(f"Settings: DPI={args.dpi}, Quality={args.quality}, Enhance={not args.no_enhance}")
    logging.info(f"Output directory: {args.output_dir.resolve()}\n")

    success = 0
    for idx, pdf_path in enumerate(pdf_paths):
        logging.info(f"[{idx+1}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            out = preprocess_pdf(
                pdf_path    = pdf_path,
                output_dir  = args.output_dir,
                dpi         = args.dpi,
                quality     = args.quality,
                enhance     = not args.no_enhance,
                overwrite   = args.overwrite,
            )
            logging.info(f"  ✓ Done → {out}\n")
            success += 1
        except Exception as e:
            logging.error(f"  ✗ Failed: {e}\n")

    logging.info(f"Preprocessing complete: {success}/{len(pdf_paths)} succeeded.")
    logging.info(f"Open files in '{args.output_dir}/' to verify quality before running gemini.py")


if __name__ == "__main__":
    main()