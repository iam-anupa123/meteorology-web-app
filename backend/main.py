"""
Meteorology Chart OCR Backend
FastAPI + OpenCV + Tesseract OCR pipeline for extracting content
from meteorological paper slips (pluviograph charts).
Handles any color scheme: grayscale, green grid, blue grid, colored.
"""

import os
import re
import hashlib
import traceback

import cv2
import numpy as np
import pytesseract
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector

# ---------------------------------------------------------------------------
# Tesseract path (Windows default install location)
# ---------------------------------------------------------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Meteorology Chart OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Anumanu",
    "database": "sys",
}


def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ===================================================================
#  IMAGE  PREPROCESSING  PIPELINE  (OpenCV)
# ===================================================================

def detect_color_scheme(image: np.ndarray) -> str:
    """Identify the dominant color scheme of the paper slip."""
    if len(image.shape) == 2:
        return "grayscale"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green grid detection (pluviograph charts)
    green_mask = cv2.inRange(hsv, (35, 25, 25), (85, 255, 255))
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    if green_ratio > 0.03:
        return "green_grid"

    # Blue grid detection
    blue_mask = cv2.inRange(hsv, (90, 25, 25), (130, 255, 255))
    blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

    if blue_ratio > 0.03:
        return "blue_grid"

    # Red grid detection
    red_mask1 = cv2.inRange(hsv, (0, 25, 25), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 25, 25), (180, 255, 255))
    red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / hsv[:, :, 0].size

    if red_ratio > 0.03:
        return "red_grid"

    return "colored"


def remove_colored_grid(image: np.ndarray, color_scheme: str) -> np.ndarray:
    """Remove the grid overlay, keeping text intact."""
    if color_scheme == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color_scheme == "green_grid":
        mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))
    elif color_scheme == "blue_grid":
        mask = cv2.inRange(hsv, (90, 20, 20), (130, 255, 255))
    elif color_scheme == "red_grid":
        m1 = cv2.inRange(hsv, (0, 20, 20), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 20, 20), (180, 255, 255))
        mask = m1 | m2
    else:
        # Generic: just convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = image.copy()
    result[mask > 0] = [255, 255, 255]
    return result


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Color-scheme aware grid removal
    2. Grayscale conversion
    3. Up-scale small images
    4. CLAHE contrast enhancement
    5. Denoising
    6. Adaptive thresholding
    """
    # Step 1 – detect colour scheme & remove grid
    color_scheme = detect_color_scheme(image)
    cleaned = remove_colored_grid(image, color_scheme)

    # Step 2 – grayscale
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if len(cleaned.shape) == 3 else cleaned

    # Step 3 – upscale if width < 2000 px
    h, w = gray.shape[:2]
    if w < 2000:
        scale = 2000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Step 4 – CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 5 – denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=12)

    # Step 6 – adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )

    # Small morphological close to join broken characters
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# ===================================================================
#  REGION  EXTRACTION  (top-to-bottom)
# ===================================================================

def split_into_regions(image: np.ndarray) -> dict:
    """
    Divide the paper slip into logical regions:
      header  – top ~15 %  (station info, dates)
      left    – left ~8 %  (vertical department label)
      graph   – central area
      footer  – bottom ~12 %  (recording details, manufacturer)
    """
    h, w = image.shape[:2]

    header_end = int(h * 0.15)
    footer_start = int(h * 0.88)
    left_end = int(w * 0.08)

    return {
        "header": image[0:header_end, :],
        "left_margin": image[header_end:footer_start, 0:left_end],
        "graph": image[header_end:footer_start, left_end:],
        "footer": image[footer_start:, :],
    }


# ===================================================================
#  OCR  HELPERS
# ===================================================================

def run_ocr(region: np.ndarray, psm: int = 6) -> str:
    """Run Tesseract on an already-preprocessed region."""
    try:
        pil_img = Image.fromarray(region)
        config = f"--psm {psm} --oem 3"
        text = pytesseract.image_to_string(pil_img, config=config)
        return text.strip()
    except Exception:
        return ""


def extract_all_text(image: np.ndarray) -> dict:
    """
    Extract text from every region of the paper slip,
    reading top-to-bottom.
    """
    regions = split_into_regions(image)

    results = {}

    # ---- HEADER (top strip – station, dates, chart info) ----
    header_bin = enhance_for_ocr(regions["header"])
    results["header_text"] = run_ocr(header_bin, psm=6)

    # ---- LEFT MARGIN (vertical text → rotate 90° CCW) ----
    left_bin = enhance_for_ocr(regions["left_margin"])
    rotated = cv2.rotate(left_bin, cv2.ROTATE_90_COUNTERCLOCKWISE)
    results["left_text"] = run_ocr(rotated, psm=6)

    # ---- FOOTER (bottom strip – totals, recorder, manufacturer) ----
    footer_bin = enhance_for_ocr(regions["footer"])
    results["footer_text"] = run_ocr(footer_bin, psm=6)

    # ---- FULL IMAGE (fallback / supplementary) ----
    full_bin = enhance_for_ocr(image)
    results["full_text"] = run_ocr(full_bin, psm=6)

    # ---- GRAPH AREA – check for handwritten notes ----
    graph_bin = enhance_for_ocr(regions["graph"])
    graph_text = run_ocr(graph_bin, psm=6)
    # Only keep if there's meaningful text (not just OCR noise)
    cleaned = re.sub(r"[^A-Za-z]", "", graph_text)
    if len(cleaned) > 5:
        results["graph_text"] = graph_text
    else:
        results["graph_text"] = ""

    return results


# ===================================================================
#  METADATA  PARSING
# ===================================================================

def parse_metadata(text_results: dict) -> dict:
    """Parse structured fields from the OCR output."""

    header = text_results.get("header_text", "")
    footer = text_results.get("footer_text", "")
    full = text_results.get("full_text", "")
    graph = text_results.get("graph_text", "")

    combined = f"{header}\n{footer}\n{full}"

    metadata = {
        "station_name": "",
        "chart_set_at": "",
        "set_date": "",
        "chart_removed_at": "",
        "removed_date": "",
        "time_on": "",
        "time_off": "",
        "duration_rainfall": "",
        "raw_text": "",
    }

    # ---- Build readable raw text (top → bottom order) ----
    raw_parts = []
    for label, key in [
        ("HEADER", "header_text"),
        ("LEFT MARGIN", "left_text"),
        ("GRAPH NOTES", "graph_text"),
        ("FOOTER", "footer_text"),
    ]:
        txt = text_results.get(key, "").strip()
        if txt:
            raw_parts.append(f"=== {label} ===\n{txt}")

    # Always include full-image pass
    full_txt = text_results.get("full_text", "").strip()
    if full_txt:
        raw_parts.append(f"=== FULL PAGE ===\n{full_txt}")

    metadata["raw_text"] = "\n\n".join(raw_parts)

    # ---- Regex field extraction ----

    # Station name
    m = re.search(
        r"STATION[:\s._]*([A-Za-z0-9/\s]+?)(?:\s{2,}|CHART|YEAR|MONTH|$)",
        combined, re.IGNORECASE,
    )
    if m:
        metadata["station_name"] = m.group(1).strip()

    # Chart set at (time)
    m = re.search(r"CHART\s*SET\s*AT[:\s._]*(\d{1,2}[:\s.]\d{2})", combined, re.IGNORECASE)
    if m:
        metadata["chart_set_at"] = m.group(1).strip()

    # Set date  (ON dd/mm/yyyy)
    m = re.search(r"ON[:\s._]*(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4})", combined, re.IGNORECASE)
    if m:
        metadata["set_date"] = m.group(1).strip()

    # Chart removed at (time)
    m = re.search(r"CHART\s*REMOVED\s*AT[:\s._]*(\d{1,2}[:\s.]?\d{0,2})", combined, re.IGNORECASE)
    if m:
        metadata["chart_removed_at"] = m.group(1).strip()

    # Removed date
    m = re.search(
        r"(?:REMOVED\s*AT|REMOVED)[:\s._]*\d{1,2}[:\s.]?\d{0,2}\s*(?:ON)?\s*(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2,4})",
        combined, re.IGNORECASE,
    )
    if m:
        metadata["removed_date"] = m.group(1).strip()

    # Time on
    m = re.search(r"TIME\s*ON[:\s._]*(\d{1,2}[:\s.]?\d{0,4}\s*HRS?)?", combined, re.IGNORECASE)
    if m and m.group(1):
        metadata["time_on"] = m.group(1).strip()

    # Time off
    m = re.search(r"TIME\s*OFF[:\s._]*(\d{1,2}[:\s.]?\d{0,4}\s*HRS?)?", combined, re.IGNORECASE)
    if m and m.group(1):
        metadata["time_off"] = m.group(1).strip()

    # Duration of rainfall
    m = re.search(r"DURATION\s*(?:OF\s*)?RAINFALL[:\s._]*([0-9hHmM\s.]+)", combined, re.IGNORECASE)
    if m:
        metadata["duration_rainfall"] = m.group(1).strip()

    return metadata


# ===================================================================
#  RAINFALL  TRACE  EXTRACTION  (OpenCV colour isolation)
# ===================================================================

def _detect_trace_mask(graph_bgr: np.ndarray) -> np.ndarray | None:
    """
    Isolate the ink-trace line from the graph area by colour.
    Handles red, blue/purple, and dark/black ink traces.
    """
    hsv = cv2.cvtColor(graph_bgr, cv2.COLOR_BGR2HSV)

    candidates = {}

    # Red ink (wide range to capture faded red too)
    m1 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
    m2 = cv2.inRange(hsv, (155, 40, 40), (180, 255, 255))
    candidates["red"] = m1 | m2

    # Blue / purple ink (widened range for purple & dark blue)
    candidates["blue"] = cv2.inRange(hsv, (85, 30, 30), (150, 255, 255))

    # Dark / black ink
    candidates["dark"] = cv2.inRange(hsv, (0, 0, 0), (180, 70, 70))

    best_name, best_count = None, 0
    for name, mask in candidates.items():
        c = int(np.sum(mask > 0))
        if c > best_count:
            best_name, best_count = name, c

    if best_count < 50:
        return None

    mask = candidates[best_name]

    # --- Morphological cleanup tuned for line traces ---
    # 1. Close small gaps along the trace
    k_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 2. Horizontal closing – bridges short horizontal gaps in the trace
    #    without merging vertically-separated blobs (e.g. text)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h)

    # 3. Remove small blobs (text characters, dots, noise)
    #    via connected-component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    # Keep only components that are wide enough to be part of the trace.
    # A real trace should span a significant fraction of the graph width.
    gh, gw = mask.shape[:2]
    min_width = gw * 0.05  # component must be at least 5% of graph width

    clean_mask = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        comp_area = stats[lbl, cv2.CC_STAT_AREA]
        # Keep if wide enough OR if area is significant
        if comp_w >= min_width or comp_area > (gh * gw * 0.002):
            clean_mask[labels == lbl] = 255

    if np.sum(clean_mask > 0) < 30:
        return mask  # fallback to unfiltered mask

    return clean_mask


def _sample_trace_value(mask: np.ndarray, x_center: int, gh: int, gw: int,
                        strip_half: int = 20) -> float | None:
    """
    Sample the trace y-position at a given x column.
    Uses a wide strip and picks the BOTTOMMOST cluster of trace pixels
    (closest to baseline = highest y = lowest rainfall value on the
    inverted y-axis of the chart).  This avoids being pulled up by
    stray text/noise pixels near the top.

    Returns the value in mm, or None if no trace pixels found.
    """
    x0 = max(0, x_center - strip_half)
    x1 = min(gw, x_center + strip_half + 1)

    col_strip = mask[:, x0:x1]
    ys = np.where(col_strip > 0)[0]

    if len(ys) == 0:
        return None

    # Use the bottommost cluster: take the bottom 40 % of detected pixels
    # This avoids stray noise/text pixels far from the trace
    ys_sorted = np.sort(ys)
    bottom_start = max(0, int(len(ys_sorted) * 0.6))
    bottom_ys = ys_sorted[bottom_start:]
    y_pos = float(np.median(bottom_ys))

    # Also compute the overall median for cross-check
    y_overall = float(np.median(ys))

    # If the bottom cluster and overall median are close, use overall (more stable)
    # If far apart, prefer bottom cluster (text contamination likely at top)
    if abs(y_pos - y_overall) < gh * 0.15:
        y_final = y_overall
    else:
        y_final = y_pos

    return y_final


def extract_rainfall_trace(image: np.ndarray) -> list[dict]:
    """
    Extract time-series rainfall data from the chart trace.
    Returns list of {time_stamp, value}.

    Improvements over v1:
      - Wide sampling strips (±20 px)
      - Connected-component filtering removes text artifacts
      - Horizontal morphology preserves trace continuity
      - Linear interpolation for any remaining gaps
    """
    h, w = image.shape[:2]

    # Crop to graph area
    top = int(h * 0.15)
    bottom = int(h * 0.88)
    left = int(w * 0.08)
    right = int(w * 0.97)
    graph = image[top:bottom, left:right]
    gh, gw = graph.shape[:2]

    if len(graph.shape) < 3:
        # Grayscale fallback
        _, binary = cv2.threshold(graph, 100, 255, cv2.THRESH_BINARY_INV)
        mask = binary
    else:
        mask = _detect_trace_mask(graph)
        if mask is None:
            return []

    # --- Chart parameters ---
    Y_MAX_MM = 10.0
    NUM_SAMPLES = 25  # one per hour boundary (08:00 → 08:00)

    # Sampling strip half-width: scale with image width
    strip_half = max(15, gw // 60)

    # --- First pass: sample raw values ---
    raw_y = []  # y pixel positions (None if missing)
    for i in range(NUM_SAMPLES):
        x = int(i / (NUM_SAMPLES - 1) * (gw - 1))
        y_val = _sample_trace_value(mask, x, gh, gw, strip_half)
        raw_y.append(y_val)

    # --- Second pass: linear interpolation for gaps ---
    for i in range(NUM_SAMPLES):
        if raw_y[i] is not None:
            continue

        # Find nearest left and right non-None values
        left_idx, left_val = None, None
        for j in range(i - 1, -1, -1):
            if raw_y[j] is not None:
                left_idx, left_val = j, raw_y[j]
                break

        right_idx, right_val = None, None
        for j in range(i + 1, NUM_SAMPLES):
            if raw_y[j] is not None:
                right_idx, right_val = j, raw_y[j]
                break

        if left_val is not None and right_val is not None:
            # Linear interpolation
            t = (i - left_idx) / (right_idx - left_idx)
            raw_y[i] = left_val + t * (right_val - left_val)
        elif left_val is not None:
            raw_y[i] = left_val  # extend last known value
        elif right_val is not None:
            raw_y[i] = right_val
        # else: stays None → will map to 0.0

    # --- Convert y-pixel positions to mm values ---
    records = []
    for i in range(NUM_SAMPLES):
        if raw_y[i] is not None:
            value = round((1.0 - raw_y[i] / gh) * Y_MAX_MM, 2)
            value = max(0.0, value)
        else:
            value = 0.0

        hour = (8 + i) % 24 or 24
        records.append({"time_stamp": f"{hour:02d}:00", "value": value})

    return records


# ===================================================================
#  API  ENDPOINTS
# ===================================================================

@app.post("/api/charts/upload")
async def upload_chart(file: UploadFile = File(...)):
    """Upload a chart image → preprocess → OCR → store."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    img_hash = compute_hash(contents)

    db = get_db()
    cursor = db.cursor()

    try:
        # Duplicate check
        cursor.execute(
            "SELECT id FROM uploaded_images WHERE image_hash = %s", (img_hash,)
        )
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="This image has already been uploaded.")

        # Decode image
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # ---------- OCR pipeline ----------
        text_results = extract_all_text(image)
        metadata = parse_metadata(text_results)
        trace_data = extract_rainfall_trace(image)

        # ---------- Persist ----------
        # Clear previous data
        cursor.execute("DELETE FROM rainfall_records")
        cursor.execute("DELETE FROM chart_metadata")

        # Metadata
        cursor.execute(
            """INSERT INTO chart_metadata
               (station_name, chart_set_at, set_date,
                chart_removed_at, removed_date,
                time_on, time_off, duration_rainfall, raw_text)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                metadata["station_name"],
                metadata["chart_set_at"],
                metadata["set_date"],
                metadata["chart_removed_at"],
                metadata["removed_date"],
                metadata["time_on"],
                metadata["time_off"],
                metadata["duration_rainfall"],
                metadata["raw_text"],
            ),
        )

        # Rainfall records
        for rec in trace_data:
            cursor.execute(
                "INSERT INTO rainfall_records (time_stamp, value) VALUES (%s, %s)",
                (rec["time_stamp"], rec["value"]),
            )

        # Image hash
        cursor.execute(
            "INSERT INTO uploaded_images (image_hash) VALUES (%s)", (img_hash,)
        )

        db.commit()

        return {
            "message": "Chart uploaded and processed successfully",
            "records": len(trace_data),
            "metadata": metadata,
            "text_extracted": bool(metadata["raw_text"]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        cursor.close()
        db.close()


@app.get("/api/charts/data")
def get_data():
    """Return all rainfall time-series records."""
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, time_stamp, value FROM rainfall_records ORDER BY id"
        )
        return cursor.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cursor.close()
        db.close()


@app.get("/api/charts/metadata")
def get_metadata():
    """Return the latest chart metadata."""
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM chart_metadata ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row if row else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cursor.close()
        db.close()


@app.delete("/api/charts/reset")
def reset_data():
    """Clear all stored data so images can be re-uploaded."""
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM rainfall_records")
        cursor.execute("DELETE FROM chart_metadata")
        cursor.execute("DELETE FROM uploaded_images")
        db.commit()
        return {"message": "All data cleared successfully"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cursor.close()
        db.close()

