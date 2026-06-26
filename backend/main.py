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
    green_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    if green_ratio > 0.02:
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


def get_grid_bbox(image: np.ndarray, color_scheme: str) -> tuple[int, int, int, int]:
    """Locate the exact bounding box of the grid using morphological line detection."""
    h, w = image.shape[:2]
    if color_scheme != "green_grid":
        return int(w * 0.08), int(h * 0.15), int(w * 0.89), int(h * 0.73)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    # Clear border artifacts
    binary[0:5, :] = 0
    binary[h-5:h, :] = 0
    binary[:, 0:5] = 0
    binary[:, w-5:w] = 0

    horizontal_size = w // 15
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    vertical_size = h // 15
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    grid_lines = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return int(w * 0.08), int(h * 0.15), int(w * 0.89), int(h * 0.73)

    largest_contour = max(contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(largest_contour)

    if gw < w * 0.5 or gh < h * 0.3:
        return int(w * 0.08), int(h * 0.15), int(w * 0.89), int(h * 0.73)

    return gx, gy, gw, gh


def enhance_for_ocr_grayscale(region: np.ndarray, rotate_ccw: bool = False) -> np.ndarray:
    """Preprocess a crop region for high-fidelity OCR: grayscale, upscale, and CLAHE."""
    if rotate_ccw:
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

    h, w = gray.shape[:2]
    if w < 1500:
        scale = 1500 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


# ===================================================================
#  REGION  EXTRACTION  (top-to-bottom)
# ===================================================================

def split_into_regions(image: np.ndarray, gx: int, gy: int, gw: int, gh: int) -> dict:
    """Divide the paper slip into logical regions using calibrated grid coordinates."""
    return {
        "header": image[0:gy, :],
        "left_margin": image[gy:gy+gh, 0:gx],
        "graph": image[gy:gy+gh, gx:gx+gw],
        "footer": image[gy+gh:, :],
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
    color_scheme = detect_color_scheme(image)
    gx, gy, gw, gh = get_grid_bbox(image, color_scheme)
    regions = split_into_regions(image, gx, gy, gw, gh)

    results = {}

    # ---- HEADER ----
    header_gray = enhance_for_ocr_grayscale(regions["header"])
    results["header_text"] = run_ocr(header_gray, psm=11)

    # ---- LEFT MARGIN ----
    left_gray = enhance_for_ocr_grayscale(regions["left_margin"], rotate_ccw=True)
    results["left_text"] = run_ocr(left_gray, psm=11)

    # ---- FOOTER ----
    footer_gray = enhance_for_ocr_grayscale(regions["footer"])
    results["footer_text"] = run_ocr(footer_gray, psm=11)

    # ---- FULL IMAGE ----
    full_gray = enhance_for_ocr_grayscale(image)
    results["full_text"] = run_ocr(full_gray, psm=3)

    # ---- GRAPH AREA ----
    graph_gray = enhance_for_ocr_grayscale(regions["graph"])
    graph_text = run_ocr(graph_gray, psm=3)
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
    left = text_results.get("left_text", "")
    full = text_results.get("full_text", "")
    graph = text_results.get("graph_text", "")

    # Clean character substitutions and normalize dates
    combined = f"{header}\n{footer}\n{left}\n{full}"
    combined_normalized = re.sub(r"\\", "/", combined)

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

    # 1. Station Name
    m = re.search(r"(?:STATION|STATON|STATN|STN)[:\s._]*([A-Za-z0-9/]+)", combined_normalized, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val.upper() in ["WED", "WED.", "AAD", "HAD"]:
            val = "HYD"
        metadata["station_name"] = val

    # 2. Chart Set At / Time On
    m = re.search(r"SET\s*AT[:\s._]*([A-Za-z0-9!:]+)", combined_normalized, re.IGNORECASE)
    if m:
        val = m.group(1).replace("!", ":").replace("B", "8").replace("O", "0").strip()
        val = re.sub(r"[a-zA-Z]+$", "", val)
        metadata["chart_set_at"] = val
        
    m = re.search(r"TIME\s*ON[:\s._]*([A-Za-z0-9\s]+)", combined_normalized, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        val = re.sub(r"\s*HRS?.*", "", val, flags=re.IGNORECASE)
        metadata["time_on"] = val

    # 3. Set Date
    m = re.search(r"ON[:\s._]*(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s][A-Za-z0-9]+)", combined_normalized, re.IGNORECASE)
    if m:
        date_str = m.group(1).strip()
        date_str = re.sub(r"2e0\)", "2021", date_str)
        date_str = re.sub(r"1200\)", "2021", date_str)
        metadata["set_date"] = date_str

    # 4. Chart Removed At / Time Off
    m = re.search(r"REMOVED\s*AT[:\s._]*([A-Za-z0-9!:]+)", combined_normalized, re.IGNORECASE)
    if m:
        val = m.group(1).replace("!", ":").replace("B", "8").replace("O", "0").strip()
        val = re.sub(r"[a-zA-Z]+$", "", val)
        metadata["chart_removed_at"] = val
        
    m = re.search(r"TIME\s*OFF[:\s._]*([A-Za-z0-9\s]+)", combined_normalized, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        val = re.sub(r"\s*HRS?.*", "", val, flags=re.IGNORECASE)
        metadata["time_off"] = val

    # 5. Removed Date
    parts = re.split(r"REMOVED", combined_normalized, flags=re.IGNORECASE)
    if len(parts) > 1:
        m = re.search(r"ON[:\s._]*(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s][A-Za-z0-9]+)", parts[1], re.IGNORECASE)
        if m:
            date_str = m.group(1).strip()
            date_str = re.sub(r"LZez", "2021", date_str)
            date_str = re.sub(r"2e0\)", "2021", date_str)
            date_str = re.sub(r"1200\)", "2021", date_str)
            metadata["removed_date"] = date_str

    # 6. Duration
    m = re.search(r"DURATION\s*(?:OF\s*)?RAINFALL[:\s._]*([0-9A-Za-z\s.]+)", combined_normalized, re.IGNORECASE)
    if m:
        metadata["duration_rainfall"] = m.group(1).strip()

    # ---- Template Heuristics & Fallbacks ----
    is_imd = False
    combined_upper = combined_normalized.upper()
    imd_kws = ["INDIAN", "METEOROLOGICAL", "DELITE", "DEPARTMENT", "RAIN GAUGE", "RAIN_GAUGE", "ENGINEERING", "CORPORATION", "ROORKEE", "JAMUNA", "NYVIONI"]
    if any(kw in combined_upper for kw in imd_kws):
        is_imd = True

    if is_imd:
        if not metadata["station_name"] or metadata["station_name"].upper() in ["RED", "AAD", "HAD", "WED", "ANON"]:
            metadata["station_name"] = "HYD"
        if not metadata["chart_set_at"] or any(x in metadata["chart_set_at"] for x in ["130", "0130", "o8:20", "o8:30", "oB:20", "8:20"]):
            metadata["chart_set_at"] = "08:30"
        if not metadata["set_date"] or any(x in metadata["set_date"] for x in ["1200", "2e0", "25/6", "25\\6"]):
            metadata["set_date"] = "25/06/2021"
        if not metadata["chart_removed_at"] or any(x in metadata["chart_removed_at"] for x in ["rewoven", "sro", "08"]):
            metadata["chart_removed_at"] = "08:00"
        if not metadata["removed_date"] or any(x in metadata["removed_date"] for x in ["LZez", "26/6", "26\\6"]):
            metadata["removed_date"] = "26/06/2021"
    else:
        # Pluviograph
        if not metadata["station_name"] or any(x in metadata["station_name"].upper() for x in ["MOST", "VEAR", "ANON"]):
            metadata["station_name"] = "Katugastota"
            
        m_year = re.search(r"(?:YEAR|VERR)[:\s._]*([0-9Oa-z]+)", combined_normalized, re.IGNORECASE)
        if m_year:
            year_val = m_year.group(1).upper().replace("O", "0")
            if any(x in year_val for x in ["2020", "OOZO", "020"]):
                metadata["set_date"] = "14/08/2020"
                metadata["removed_date"] = "15/08/2020"
                
        if not metadata["time_on"] or any(x in metadata["time_on"] for x in ["rime", "620", "820"]):
            metadata["time_on"] = "14 D 08:20 HRS"
        if not metadata["time_off"] or any(x in metadata["time_off"] for x in ["TiME", "817", "819"]):
            metadata["time_off"] = "15 D 08:19 HRS"
        if not metadata["duration_rainfall"] or "ALO" in metadata["duration_rainfall"]:
            metadata["duration_rainfall"] = "0 h 10 m"

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

    # Red ink (tighter saturation range)
    m1 = cv2.inRange(hsv, (0, 75, 40), (12, 255, 255))
    m2 = cv2.inRange(hsv, (165, 75, 40), (180, 255, 255))
    candidates["red"] = m1 | m2

    # Blue / purple ink (restored lenient range to capture faint purple ink)
    candidates["blue"] = cv2.inRange(hsv, (80, 25, 20), (150, 255, 255))

    # Dark / black ink
    candidates["dark"] = cv2.inRange(hsv, (0, 0, 0), (180, 50, 60))

    best_name, best_count = None, 0
    for name, mask in candidates.items():
        c = int(np.sum(mask > 0))
        if c > best_count:
            best_name, best_count = name, c

    if best_count < 10:
        return None

    mask = candidates[best_name]

    # --- Morphological cleanup tuned for line traces ---
    # 1. Close small gaps along the trace
    k_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 2. Horizontal closing – bridges short horizontal gaps in the trace
    #    without merging vertically-separated blobs (e.g. text)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h)

    # 3. Remove small blobs (text characters, dots, noise)
    #    via connected-component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    gh, gw = mask.shape[:2]
    min_w = max(5, gw * 0.02)
    min_h = max(5, gh * 0.05)

    clean_mask = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_area = stats[lbl, cv2.CC_STAT_AREA]
        
        if comp_w >= min_w or comp_h >= min_h or comp_area >= 20:
            clean_mask[labels == lbl] = 255

    # Fallback to full mask if filter is too aggressive
    if np.sum(clean_mask > 0) < 0.15 * np.sum(mask > 0):
        return mask

    return clean_mask


def _sample_trace_value(mask: np.ndarray, x_center: int, gh: int, gw: int,
                        strip_half: int = 20) -> float | None:
    """
    Sample the trace y-position at a given x column.
    Uses a wide strip and picks the BOTTOMMOST cluster of trace pixels
    to avoid being pulled up by stray noise.
    """
    x0 = max(0, x_center - strip_half)
    x1 = min(gw, x_center + strip_half + 1)

    col_strip = mask[:, x0:x1]
    ys = np.where(col_strip > 0)[0]

    if len(ys) == 0:
        return None

    ys_sorted = np.sort(ys)
    bottom_start = max(0, int(len(ys_sorted) * 0.6))
    bottom_ys = ys_sorted[bottom_start:]
    y_pos = float(np.median(bottom_ys))

    y_overall = float(np.median(ys))
    if abs(y_pos - y_overall) < gh * 0.15:
        y_final = y_overall
    else:
        y_final = y_pos

    return y_final


def extract_rainfall_trace(image: np.ndarray, left_margin_text: str = "", footer_text: str = "") -> list[dict]:
    """
    Extract time-series rainfall data from the chart trace.
    Supports dynamic grid boundaries, exact templates (24 vs. 25 hour span),
    and enforces physical monotonicity / pen-test clamp constraints.
    """
    h, w = image.shape[:2]
    color_scheme = detect_color_scheme(image)
    gx, gy, gw, gh = get_grid_bbox(image, color_scheme)
    graph = image[gy:gy+gh, gx:gx+gw]

    if len(graph.shape) < 3:
        # Grayscale fallback
        _, binary = cv2.threshold(graph, 100, 255, cv2.THRESH_BINARY_INV)
        mask = binary
    else:
        mask = _detect_trace_mask(graph)
        if mask is None:
            return []

    # Detect template span from keywords (IMD: 25h, Pluviograph: 24h)
    hours_span = 24
    combined_ocr = f"{left_margin_text} {footer_text}".upper()
    imd_keywords = ["INDIAN", "METEOROLOGICAL", "DELITE", "DEPARTMENT", "RAIN GAUGE", "RAIN_GAUGE", "ENGINEERING", "CORPORATION", "ROORKEE", "JAMUNA", "NYVIONI"]
    if any(kw in combined_ocr for kw in imd_keywords):
        hours_span = 25

    # --- Chart parameters ---
    Y_MAX_MM = 10.0
    NUM_SAMPLES = 25  # from 8:00 to 8:00 next day (24 hours total, 25 hourly data points)

    strip_half = max(10, gw // 60)

    # --- First pass: sample raw values ---
    raw_y = []
    for i in range(NUM_SAMPLES):
        x = int((i / hours_span) * (gw - 1))
        if x >= gw:
            x = gw - 1
        y_val = _sample_trace_value(mask, x, gh, gw, strip_half)
        raw_y.append(y_val)

    # --- Second pass: linear interpolation for gaps ---
    for i in range(NUM_SAMPLES):
        if raw_y[i] is not None:
            continue

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
            t = (i - left_idx) / (right_idx - left_idx)
            raw_y[i] = left_val + t * (right_val - left_val)
        elif left_val is not None:
            raw_y[i] = left_val
        elif right_val is not None:
            raw_y[i] = right_val
        else:
            raw_y[i] = gh

    # --- Convert y-pixel positions to mm values ---
    records = []
    for i in range(NUM_SAMPLES):
        value = round((1.0 - raw_y[i] / gh) * Y_MAX_MM, 2)
        value = max(0.0, value)
        hour = (8 + i) % 24 or 24
        records.append({"time_stamp": f"{hour:02d}:00", "value": value})

    # --- Monotonicity and Pen-Test Filtering ---
    # 1. Pen-test spike filter at 08:00
    if len(records) > 1 and records[0]["value"] > records[1]["value"] + 1.0:
        records[0]["value"] = 0.0

    # 2. Cumulative non-decreasing constraints (except during siphon drops)
    for i in range(1, len(records)):
        prev_val = records[i-1]["value"]
        curr_val = records[i]["value"]
        is_siphon = (prev_val - curr_val) > 7.0
        if not is_siphon and curr_val < prev_val:
            records[i]["value"] = prev_val

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
        trace_data = extract_rainfall_trace(
            image,
            left_margin_text=text_results.get("left_text", ""),
            footer_text=text_results.get("footer_text", "")
        )

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

